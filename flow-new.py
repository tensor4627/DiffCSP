import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal



from ase import Atoms
from ase.neighborlist import neighbor_list
from diffcsp.pl_modules.miaonet.liqmiao import EmoMiaoNet,TimeEmbedding
from hotpp.layer.embedding import AtomicEmbedding
from hotpp.layer.radial import BesselPoly
from hotpp.layer.cutoff import PolynomialCutoff
def collect_data(datas):
    elem = datas[0]
    coll_batch = {}

    for key in elem:
        if key not in ["idx_i", "idx_j"]:
            coll_batch[key] = torch.cat([d[key] for d in datas], dim=0)

    # idx_i and idx_j should to be converted like
    # [0, 0, 1, 1] + [0, 0, 1, 2] -> [0, 0, 1, 1, 2, 2, 3, 4]
    for key in ["idx_i", "idx_j"]:
        coll_batch[key] = torch.cat(
            [
                datas[i][key] + torch.sum(coll_batch["n_atoms"][:i])
                for i in range(len(datas))
            ],
            dim=0,
        )

    coll_batch["batch"] = torch.repeat_interleave(
        torch.arange(len(datas), device=coll_batch["n_atoms"].device),
        repeats=coll_batch["n_atoms"].to(torch.long),
        dim=0,
    )
    return coll_batch


MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        print(pred_x[0],tar_x[0])
        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss



class CSPFlow(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_steps = self.hparams.time_steps
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    @staticmethod
    def wrapped_distance_vector(start_fpos,end_fpos):
        dis = end_fpos-start_fpos
        #return dis-torch.round(dis)
        return (dis-0.5)%1-0.5
    
    def get_static_noise(self,scaled_positions,cells):
        scaled_positions_noise = torch.rand_like(scaled_positions)
        a_low,a_high = 2.4,12.8
        alpha_low,alpha_high = 60,120
        low = torch.tensor([a_low,a_low,a_low,alpha_low,alpha_low,alpha_low],device=cells.device,dtype=cells.dtype)
        high = torch.tensor([a_high,a_high,a_high,alpha_high,alpha_high,alpha_high],device=cells.device,dtype=cells.dtype)
        cells_noise_6d = low + (high - low) * torch.rand(size=(cells.shape[0],low.shape[0]),device=cells.device,dtype=cells.dtype)
        cells_noise = self.lattice_from_parameters(cells_noise_6d[:,0],
                                                   cells_noise_6d[:,1],
                                                   cells_noise_6d[:,2],
                                                   cells_noise_6d[:,3],
                                                   cells_noise_6d[:,4],
                                                   cells_noise_6d[:,5])
        return scaled_positions_noise,cells_noise       


    @staticmethod
    def lattice_from_parameters(a, b, c, alpha, beta, gamma, degrees=True):
        if degrees:
            alpha = torch.deg2rad(alpha)
            beta  = torch.deg2rad(beta)
            gamma = torch.deg2rad(gamma)
        cos_alpha = torch.cos(alpha)
        cos_beta  = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        a1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
        a2 = torch.stack([
            b * cos_gamma,
            b * sin_gamma,
            torch.zeros_like(b)
        ], dim=-1)
        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = c * torch.sqrt(
            1 - cos_beta**2
            - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2
        )
        a3 = torch.stack([cx, cy, cz], dim=-1)
        lattice = torch.stack([a1, a2, a3], dim=-2)
        return lattice


    def forward(self, batch):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        #rand_x,rand_l = torch.rand_like(frac_coords),torch.rand_like(lattices)
        rand_x,rand_l = self.get_static_noise(frac_coords,lattices)
        input_lattice = rand_l+(lattices-rand_l)*times.view(-1,1,1)/self.time_steps
        input_frac_coords = rand_x + self.wrapped_distance_vector(rand_x,frac_coords)*(times.repeat_interleave(batch.num_atoms)[:, None])/self.time_steps

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l_velocity, pred_x_velocity = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        lattice_inv = torch.inverse(rand_l)
        #loss_lattice = F.mse_loss(pred_l_velocity, (lattices-rand_l))
        loss_lattice = F.mse_loss(torch.bmm(lattice_inv,pred_l_velocity),torch.bmm(lattice_inv,(lattices-rand_l)))
        loss_coord = F.mse_loss(pred_x_velocity, self.wrapped_distance_vector(rand_x,frac_coords))
        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss


class HotPPFlow(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cutoff = 3.0
        t_max = 1000
        self.model = EmoMiaoNet(
            cutoff=3.0,
            t_max= 1000,
            betas=[1.0,],
            sigmas=[1.0,],
            embedding_layer=AtomicEmbedding([6],n_channel=64),
            time_embedding=TimeEmbedding(64,t_max),
            radial_fn=BesselPoly(r_max=cutoff,n_max=8,cutoff_fn=PolynomialCutoff(cutoff=cutoff,p=5)),
            n_layers=5,
            max_r_way=[2]*5,
            max_out_way=[2]*5,
            output_dim=[64]*5,
            activate_fn='silu',
            norm_factor = 18.0
        )
        self.cutoff=cutoff
        self.t_max = t_max
        self.time_steps = t_max
    @staticmethod
    def get_batch_data(n_atoms, atomic_numbers, scaled_positions, cells, cutoff, device):
        datas = []
        n_frames = n_atoms.shape[0]
        split_idx = [0] + list(np.cumsum(n_atoms))
        for i in range(n_frames):
            atoms = Atoms(
                numbers=atomic_numbers[split_idx[i] : split_idx[i + 1]],
                scaled_positions=scaled_positions[split_idx[i] : split_idx[i + 1]],
                cell=cells[i],
                pbc=True,
            )
            idx_i, idx_j, offset = neighbor_list(
                "ijS", atoms, cutoff, self_interaction=False
            )
            cell = atoms.cell[:]
            inv_cell = atoms.cell.reciprocal()[:].T
            offset = np.array(offset) @ cell
            datas.append(
                {
                    "atomic_number": torch.tensor(
                        atoms.numbers, dtype=torch.long, device=device
                    ),
                    "idx_i": torch.tensor(idx_i, dtype=torch.long, device=device),
                    "idx_j": torch.tensor(idx_j, dtype=torch.long, device=device),
                    "coordinate": torch.tensor(
                        atoms.positions, dtype=torch.float32, device=device
                    ),
                    "cell": torch.tensor(
                        cell.reshape(1, 3, 3), dtype=torch.float32, device=device
                    ),
                    "inv_cell": torch.tensor(
                        inv_cell.reshape(1, 3, 3), dtype=torch.float32, device=device
                    ),
                    "n_atoms": torch.tensor([len(atoms)], dtype=torch.long, device=device),
                    "offset": torch.tensor(offset, dtype=torch.float32, device=device),
                    "scaling":torch.eye(3, dtype=torch.float32,device =device).view(1, 3, 3)
                }
            )
        batch_data  = collect_data(datas)
        batch_data["coordinate"].requires_grad_()
        batch_data["scaling"].requires_grad_()

        idx_m = batch_data['batch'].to(device)
        idx_i = batch_data['idx_i'].to(device)
        batch_data['coordinate'] = torch.matmul(
            batch_data['coordinate'][:, None, :], batch_data['scaling'][idx_m]
        ).squeeze(1)
        batch_data['offset'] = torch.matmul(
            batch_data['offset'][:, None, :], batch_data['scaling'][idx_m][idx_i]
        ).squeeze(1)

        return batch_data

    def uniform_sample_t(self,batch_size, device):
        ts = np.random.choice(np.arange(1, self.t_max+1), batch_size)
        return torch.from_numpy(ts).to(device)
    def get_static_noise(self,scaled_positions,cells):
        scaled_positions_noise = torch.rand_like(scaled_positions)
        a_low,a_high = 2.4,12.8
        alpha_low,alpha_high = 60,120
        low = torch.tensor([a_low,a_low,a_low,alpha_low,alpha_low,alpha_low],device=cells.device,dtype=cells.dtype)
        high = torch.tensor([a_high,a_high,a_high,alpha_high,alpha_high,alpha_high],device=cells.device,dtype=cells.dtype)
        cells_noise_6d = low + (high - low) * torch.rand(size=(cells.shape[0],low.shape[0]),device=cells.device,dtype=cells.dtype)
        cells_noise = self.lattice_from_parameters(cells_noise_6d[:,0],
                                                   cells_noise_6d[:,1],
                                                   cells_noise_6d[:,2],
                                                   cells_noise_6d[:,3],
                                                   cells_noise_6d[:,4],
                                                   cells_noise_6d[:,5])
        return scaled_positions_noise,cells_noise


    @staticmethod
    def lattice_from_parameters(a, b, c, alpha, beta, gamma, degrees=True):
        if degrees:
            alpha = torch.deg2rad(alpha)
            beta  = torch.deg2rad(beta)
            gamma = torch.deg2rad(gamma)
        cos_alpha = torch.cos(alpha)
        cos_beta  = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        a1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
        a2 = torch.stack([
            b * cos_gamma,
            b * sin_gamma,
            torch.zeros_like(b)
        ], dim=-1)
        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = c * torch.sqrt(
            1 - cos_beta**2
            - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2
        )
        a3 = torch.stack([cx, cy, cz], dim=-1)
        lattice = torch.stack([a1, a2, a3], dim=-2)
        return lattice

    @staticmethod
    def wrapped_distance_vector(start_fpos,end_fpos):
        dis = end_fpos-start_fpos
        return dis-torch.round(dis)

    def forward(self, batch):
        batch_size = batch.num_graphs
        times = self.uniform_sample_t(batch_size, batch.frac_coords.device)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords
        #rand_x,rand_l = torch.rand_like(frac_coords),6*torch.rand_like(lattices)
        rand_x,rand_l = self.get_static_noise(frac_coords,lattices)
        input_lattice = rand_l+(lattices-rand_l)*times.view(-1,1,1)/self.time_steps
        #input_lattice = lattices
        input_frac_coords = rand_x + self.wrapped_distance_vector(rand_x,frac_coords)*(times.repeat_interleave(batch.num_atoms)[:, None])/self.time_steps
        atomic_numbers = torch.ones_like(frac_coords)[:,0]*6
        batch_data = self.get_batch_data(
            batch.num_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            input_frac_coords.detach().clone().cpu().numpy(),
            input_lattice.detach().clone().cpu().numpy(),
            float(self.cutoff),
            device=frac_coords.device,
        )


        batch_data = self.model.model_predict(batch_data,times)
        pred_x_velocity = batch_data["frac_velocity"]
        pred_d_velocity = batch_data["deform_velocity"]
        pred_l_velocity = torch.bmm(rand_l,pred_d_velocity)
        loss_lattice = F.mse_loss(pred_l_velocity, (lattices-rand_l))
        loss_coord = F.mse_loss(pred_x_velocity, self.wrapped_distance_vector(rand_x,frac_coords))

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss

    
