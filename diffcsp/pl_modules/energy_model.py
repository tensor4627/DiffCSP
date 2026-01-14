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
from torch.autograd import grad
from tqdm import tqdm
from hotpp.utils import (
    expand_para,
    find_distances,
    _scatter_add,
    _scatter_mean,
    EnvPara,
)
from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)
MAX_ATOMIC_NUM=100

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal,get_static_noise
from scipy.optimize import linear_sum_assignment

import pdb


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

def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)


class CSPEnergy(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.time_dim, pred_scalar = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)

        if not hasattr(self.hparams, 'update_type'):
            self.update_type = True
        else:
            self.update_type = self.hparams.update_type      


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
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        if self.update_type:
            atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        else:
            atom_type_probs = gt_atom_types_onehot
            
            
        pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        loss_energy = F.l1_loss(pred_e, batch.y)


        loss = loss_energy

        return {
            'loss' : loss,
        }

    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio = 1.0, step_lr = 1e-5, aug = 1.0):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        update_type = self.update_type


        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device) if update_type else F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()


        
        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t if update_type else atom_types_onehot

        else:
            time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            if self.hparams.latent_dim > 0:            
                time_emb = torch.cat([time_emb, z], dim = -1)

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t
            
            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_T


            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)


                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1

                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_T, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_x, grad_l = grad(pred_e, [x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = t_T


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):
        
        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim = -1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta



    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):

        pred_atom_probs = F.softmax(pred_atom_types, dim = -1)

        atom_probs_0 = F.one_hot(target_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)

        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat, 
            target=theta, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):
        
        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:,types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss},
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

        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        return log_dict, loss

    
class CSPEnergyMatching(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_steps = self.hparams.timesteps
        self.learning_stage = self.hparams.learning_stage
        if self.learning_stage == "energy":
            self.dt = self.hparams.dt
            self.langevin_steps=self.hparams.langevin_steps
            self.lambda_cd = self.hparams.lambda_cd
        else:
            self.dt = None
            self.langevin_steps=None
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        # for test
        self.i = 0

    @staticmethod
    def wrapped_distance_vector(start_fpos,end_fpos):
        dis = end_fpos-start_fpos
        return dis-torch.round(dis)
    def get_static_noise(self,scaled_positions,cells,mode="uni"):
        scaled_positions_noise = torch.rand_like(scaled_positions)
        if mode == "uni":
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
        elif mode == "log":
            loc,scale = 1.3595351865425995,0.1277552273430223
            cell_l = torch.exp(scale*torch.randn(size=(cells.shape[0],3),device=cells.device,dtype=cells.dtype)+loc)
            cell_a = 60+60*torch.rand(size=(cells.shape[0],3),device=cells.device,dtype=cells.dtype)
            cells_noise = self.lattice_from_parameters(cell_l[:,0],
                                                    cell_l[:,1],
                                                    cell_l[:,2],
                                                    cell_a[:,0],
                                                    cell_a[:,1],
                                                    cell_a[:,2])
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
    def rot_tril(H):
        Q, R = torch.linalg.qr(H.transpose(-2, -1))

    # Step 2: enforce Q ∈ SO(3) (det = +1)
        detQ = torch.det(Q)                         # (B,)
        signQ = detQ.sign().view(-1, 1, 1)          # (B,1,1)
        Q = Q * signQ                               # flip one column if needed

    # Step 3: construct lower-triangular lattice
    # H = R^T Q^T  ⇒  Q^T H = R^T
        L = R.transpose(-2, -1)
        R_rot = Q.transpose(-2, -1)

    # Step 4: enforce positive diagonal of L
        diag = torch.diagonal(L, dim1=-2, dim2=-1)  # (B,3)
        diag_sign = diag.sign()
        diag_sign = torch.where(
            diag_sign == 0,
            torch.ones_like(diag_sign),
            diag_sign,
        )

        D = torch.diag_embed(diag_sign)              # (B,3,3)

        L = D @ L
        R_rot = D @ R_rot

        return L, R_rot


    def forward(self, batch):
        if self.learning_stage == "flow":
            return self.flow(batch)
        else:
            return self.energy_matching(batch)

    def flow(self,batch):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        #rand_x,rand_l = torch.rand_like(frac_coords),torch.rand_like(lattices)
        rand_x,rand_l = self.get_static_noise(frac_coords,lattices,mode="log")
        lattices,_ = self.rot_tril(lattices)
        input_lattice = rand_l+(lattices-rand_l)*times.view(-1,1,1)/self.time_steps
        input_frac_coords = rand_x + self.wrapped_distance_vector(rand_x,frac_coords)*(times.repeat_interleave(batch.num_atoms)[:, None])/self.time_steps

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        #pred_e = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        with torch.enable_grad():
            with RequiresGradContext(input_frac_coords, input_lattice, requires_grad=True):
                pred_e = (self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch))
                # grad_f, grad_l = grad(pred_e, [input_frac_coords,input_lattice], grad_outputs = grad_outputs,create_graph=True,allow_unused=True)
                grad_f, grad_l = grad([pred_e.sum()], [input_frac_coords,input_lattice],create_graph=True,allow_unused=True)
        loss_lattice = F.mse_loss(torch.tril(-grad_l), torch.tril(lattices-rand_l))
        self.i+=1
        grad_x = (grad_f.view(-1,1,3)@input_lattice.detach().transpose(-1,-2).repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)
        loss_coord = F.mse_loss((-grad_x), self.wrapped_distance_vector(rand_x,frac_coords))

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_energy': torch.zeros_like(loss_coord)
        }

    def get_forces(self,pos,cell,batch,time_emb):
        with torch.enable_grad():
            with RequiresGradContext(pos, cell, requires_grad=True):
                pred_e = (self.decoder(time_emb, batch.atom_types, pos,cell, batch.num_atoms, batch.batch))
                # grad_f, grad_l = grad(pred_e, [input_frac_coords,input_lattice], grad_outputs = grad_outputs,create_graph=True,allow_unused=True)
                grad_f, grad_l = grad([pred_e.sum()], [pos,cell],create_graph=True,allow_unused=True)
                # grad_x = (grad_f.view(-1,1,3)@cell.detach().transpose(-1,-2).repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)
        return -grad_f,-grad_l,pred_e
    
    def langevin_step(self,pos,cell,batch,time_emb,step_size,std):
        tpos = pos.clone().detach()
        tcell = cell.clone().detach()
        fpos_forces,lattice_forces,_ = self.get_forces(tpos,tcell,batch,time_emb)
        pos = pos + step_size * fpos_forces + std * torch.randn_like(pos)
        cell=cell + step_size * lattice_forces + std * torch.randn_like(cell)
        return pos,cell
    
    def energy_matching(self,batch):
        flow_loss = self.flow(batch)
        batch_size = batch.num_graphs
        # times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        rand_times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        times = torch.ones_like(rand_times)*rand_times.max()
        time_emb = self.time_embedding(times)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        # rand_x,rand_l = self.get_static_noise(frac_coords,lattices,mode="log")#negative sampling
        lattices,_ = self.rot_tril(lattices)

        l_pos = frac_coords.clone().detach()
        l_cell = lattices.clone().detach()
        for i in range(self.langevin_steps):
            l_pos,l_cell = self.langevin_step(l_pos.detach(),l_cell.detach(),batch,time_emb,step_size=self.dt,std=(2*self.dt)**0.5)
        _,_,neg_e = self.get_forces(l_pos,l_cell,batch,time_emb)
        _,_,pos_e = self.get_forces(frac_coords,lattices,batch,time_emb)
        energy_loss = torch.mean(pos_e)-torch.mean(neg_e)
        flow_loss["loss_energy"] = energy_loss
        flow_loss["loss"]+=energy_loss
        return flow_loss
    
    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio = 1.0, step_lr = 1e-5, aug = 1.0):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        update_type = self.update_type


        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device) if update_type else F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()


        
        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t if update_type else atom_types_onehot

        else:
            time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            if self.hparams.latent_dim > 0:            
                time_emb = torch.cat([time_emb, z], dim = -1)

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t
            
            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_T


            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)


                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1

                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_T, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_x, grad_l = grad(pred_e, [x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = t_T


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):
        
        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim = -1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta



    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):

        pred_atom_probs = F.softmax(pred_atom_types, dim = -1)

        atom_probs_0 = F.one_hot(target_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)

        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat, 
            target=theta, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):
        
        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:,types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if batch_idx==0:
            self.i+=1
        if self.i>=3000:
            self.learning_stage="energy"
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'energy_loss': output_dict.get('loss_energy', torch.tensor(0.0))},
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
        loss_ene=output_dict['loss_energy']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_ene_loss': loss_ene
        }

        return log_dict, loss

