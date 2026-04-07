import math, copy
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
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

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal,get_static_noise,soften_coordinates_piecewise,wrap_coordinates,matrix_log,extract_spd
from scipy.optimize import linear_sum_assignment

import pdb
import ase
from ase import Atoms
from ase.io import write


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
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss","interval":"epoch","frequency":1}


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
        self.log(
            "val_loss",
            log_dict["val_loss"],
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

    
class CSPEnergyMatchingT(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.flow_beta_scheduler = hydra.utils.instantiate(self.hparams.flow_beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.lattice_noise_mode = self.hparams.lattice_noise_mode
        self.time_dim = self.hparams.time_dim
        self.time_steps = self.hparams.timesteps
        self.flow_time_steps = self.hparams.flow_timesteps
        self.learning_stage = self.hparams.learning_stage
        self.dt = self.hparams.dt
        self.langevin_steps=self.hparams.langevin_steps
        self.lambda_cd = self.hparams.lambda_cd
        self.flow_warm_epochs = self.hparams.flow_warm_epochs
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        # for test
        self.i = 0
        self._flow_spike_dump_count = 0

    @rank_zero_only
    def _dump_anomalous_force_structures(
        self,
        batch,
        times,
        input_frac_coords,
        input_lattice,
        pred_f,
        target_f,
        pred_l,
        target_l,
        loss_coord,
    ):
        pred_abs_threshold = float(getattr(self.hparams, "debug_pred_force_abs_threshold", 2.0))
        max_atoms_per_crystal = int(getattr(self.hparams, "debug_max_atoms_per_crystal", 24))
        max_crystals_per_batch = int(getattr(self.hparams, "debug_max_crystals_per_batch", 4))
        max_total_dumps = int(getattr(self.hparams, "debug_max_total_dumps", 2000))
        if self._flow_spike_dump_count >= max_total_dumps:
            return

        pred_f_detach = pred_f.detach()
        target_f_detach = target_f.detach()
        pred_l_detach = pred_l.detach()
        target_l_detach = target_l.detach()
        frac_detach = input_frac_coords.detach()
        lattice_detach = input_lattice.detach()
        atom_types_detach = batch.atom_types.detach()
        num_atoms_list = batch.num_atoms.detach().tolist()
        times_detach = times.detach()

        abs_max_component = pred_f_detach.abs().max(dim=-1).values
        abnormal_mask = abs_max_component > pred_abs_threshold
        abnormal_idx_global = torch.nonzero(abnormal_mask, as_tuple=False).flatten()
        if abnormal_idx_global.numel() == 0:
            return

        debug_root = Path(PROJECT_ROOT) / "debug_flow_spikes"
        debug_root.mkdir(parents=True, exist_ok=True)

        atom_start = 0
        dumped = 0
        for crystal_idx, n_atoms in enumerate(num_atoms_list):
            if dumped >= max_crystals_per_batch or self._flow_spike_dump_count >= max_total_dumps:
                break
            atom_end = atom_start + n_atoms
            crystal_abnormal_mask = abnormal_mask[atom_start:atom_end]
            if not crystal_abnormal_mask.any():
                atom_start = atom_end
                continue

            local_abnormal_idx = torch.nonzero(crystal_abnormal_mask, as_tuple=False).flatten()
            local_abs = abs_max_component[atom_start:atom_end][local_abnormal_idx]
            topk = min(max_atoms_per_crystal, local_abnormal_idx.numel())
            topk_local_order = torch.topk(local_abs, k=topk, largest=True).indices
            local_abnormal_idx = local_abnormal_idx[topk_local_order]

            crystal_frac = frac_detach[atom_start:atom_end] % 1.0
            crystal_cell = lattice_detach[crystal_idx]
            crystal_numbers = atom_types_detach[atom_start:atom_end].long().cpu().numpy()

            atoms = Atoms(
                numbers=crystal_numbers,
                scaled_positions=crystal_frac.cpu().numpy(),
                cell=crystal_cell.cpu().numpy(),
                pbc=True,
            )
            atoms.arrays["pred_f"] = pred_f_detach[atom_start:atom_end].cpu().numpy()
            atoms.arrays["target_f"] = target_f_detach[atom_start:atom_end].cpu().numpy()
            atoms.arrays["abnormal_force_atom"] = crystal_abnormal_mask.long().cpu().numpy()
            atoms.info["epoch"] = int(self.current_epoch)
            atoms.info["global_step"] = int(self.global_step)
            atoms.info["learning_stage"] = str(self.learning_stage)
            atoms.info["loss_coord"] = float(loss_coord.detach().item())
            atoms.info["interp_time"] = float(times_detach[crystal_idx].item())
            atoms.info["pred_lattice_force"] = pred_l_detach[crystal_idx].cpu().numpy().tolist()
            atoms.info["target_lattice_force"] = target_l_detach[crystal_idx].cpu().numpy().tolist()
            atoms.info["abnormal_atom_local_indices"] = local_abnormal_idx.cpu().numpy().tolist()

            file_stem = (
                f"epoch{int(self.current_epoch):04d}_step{int(self.global_step):08d}"
                f"_c{crystal_idx:03d}_t{int(times_detach[crystal_idx].item()):04d}"
            )
            out_path = debug_root / f"{file_stem}.extxyz"
            write(str(out_path), atoms, format="extxyz")

            detail_path = debug_root / f"{file_stem}.txt"
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write(f"loss_coord: {float(loss_coord.detach().item()):.8e}\n")
                f.write(f"interp_time: {float(times_detach[crystal_idx].item()):.6f}\n")
                f.write(f"crystal_idx: {crystal_idx}\n")
                f.write(f"num_atoms: {n_atoms}\n")
                f.write(f"pred_lattice_force:\n{pred_l_detach[crystal_idx].cpu().numpy()}\n")
                f.write(f"target_lattice_force:\n{target_l_detach[crystal_idx].cpu().numpy()}\n")
                f.write("abnormal_atoms(local_idx, pred_f, target_f):\n")
                for local_idx in local_abnormal_idx.cpu().tolist():
                    global_idx = atom_start + local_idx
                    f.write(
                        f"{local_idx:4d} "
                        f"pred={pred_f_detach[global_idx].cpu().numpy()} "
                        f"target={target_f_detach[global_idx].cpu().numpy()}\n"
                    )

            print(
                f"[flow-debug] dumped anomalous structure to {out_path} "
                f"(abnormal_atoms={local_abnormal_idx.numel()}, threshold={pred_abs_threshold:.2f})"
            )
            dumped += 1
            self._flow_spike_dump_count += 1
            atom_start = atom_end

        if dumped == 0:
            print(
                f"[flow-debug] no crystal-level dump, but found "
                f"{abnormal_idx_global.numel()} abnormal atoms globally"
            )

    def epsilon_strategy(self,eps,step):
        now_time = self.dt*step
        if now_time<self.tau:
            return 0.
        elif now_time>=self.tau and now_time<1.:
            return eps*(now_time - self.tau)/(1.-self.tau)
        else:
            return eps


    @staticmethod
    def wrapped_distance_vector(start_fpos,end_fpos):
        dis = end_fpos-start_fpos
        return dis-torch.round(dis)

    @staticmethod
    def _tensor_stats(name, tensor):
        if tensor is None:
            return f"{name}: None"
        flat = tensor.detach().reshape(-1).float()
        finite = torch.isfinite(flat)
        if finite.any():
            f = flat[finite]
            p95 = torch.quantile(f.abs(), 0.95)
            return (
                f"{name}: mean={f.mean().item():.4e}, "
                f"abs_mean={f.abs().mean().item():.4e}, "
                f"abs_max={f.abs().max().item():.4e}, "
                f"p95_abs={p95.item():.4e}, "
                f"norm={torch.linalg.norm(f).item():.4e}, "
                f"nan={torch.isnan(flat).sum().item()}, "
                f"inf={torch.isinf(flat).sum().item()}"
            )
        return (
            f"{name}: all_non_finite, "
            f"nan={torch.isnan(flat).sum().item()}, "
            f"inf={torch.isinf(flat).sum().item()}"
        )

    def _debug_flow_spike(
        self,
        batch,
        times,
        input_frac_coords,
        input_lattice,
        lattices,
        rand_l,
        grad_f,
        grad_l,
        target_f,
        loss_coord,
    ):
        num_atoms_f = batch.num_atoms.float()
        det_in = torch.det(input_lattice.detach())
        delta_l = (lattices - rand_l).detach()
        grad_ratio = torch.linalg.norm(grad_f.detach()) / (torch.linalg.norm(target_f.detach()) + 1e-8)
        pred_f = -grad_f
        pred_l = -grad_l
        target_l = delta_l

        print(f"[flow-debug] epoch={self.current_epoch}, stage={self.learning_stage}, loss_coord={loss_coord.item():.4e}")
        print(
            f"[flow-debug] times: min={times.min().item()}, "
            f"mean={times.float().mean().item():.2f}, max={times.max().item()}"
        )
        print(
            f"[flow-debug] num_atoms: min={num_atoms_f.min().item():.0f}, "
            f"mean={num_atoms_f.mean().item():.2f}, max={num_atoms_f.max().item():.0f}"
        )
        print(self._tensor_stats("target_f", target_f))
        print(self._tensor_stats("pred_f(-grad_f)", pred_f))
        print(self._tensor_stats("grad_l", grad_l))
        print(self._tensor_stats("det(input_lattice)", det_in))
        print(self._tensor_stats("delta_lattice(l-rand)", delta_l))
        print(f"[flow-debug] grad_f/target_f norm ratio={grad_ratio.item():.4e}")

        self._dump_anomalous_force_structures(
            batch=batch,
            times=times,
            input_frac_coords=input_frac_coords,
            input_lattice=input_lattice,
            pred_f=pred_f,
            target_f=target_f,
            pred_l=pred_l,
            target_l=target_l,
            loss_coord=loss_coord,
        )

    def get_static_noise(self,scaled_positions,cells,mode=1):
        scaled_positions_noise = torch.rand_like(scaled_positions)
        mode = self.lattice_noise_mode
        if mode == 2:
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
        elif mode == 3:
            loc,scale = 1.3595351865425995,0.1277552273430223
            cell_l = torch.exp(scale*torch.randn(size=(cells.shape[0],3),device=cells.device,dtype=cells.dtype)+loc)
            cell_a = 60+60*torch.rand(size=(cells.shape[0],3),device=cells.device,dtype=cells.dtype)
            cells_noise = self.lattice_from_parameters(cell_l[:,0],
                                                    cell_l[:,1],
                                                    cell_l[:,2],
                                                    cell_a[:,0],
                                                    cell_a[:,1],
                                                    cell_a[:,2])
        elif mode == 1:
            cells_noise = torch.rand_like(cells)
        elif mode == 0:
            cells_noise = cells.detach().clone()

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

    @staticmethod
    def unwrapped_mse(pred, target):
        """
        pred:   模型预测的梯度, (B, N, 3), 值域 R^3
        target: 测地线速度,    (B, N, 3), 在 (-0.5, 0.5]
        
        将 target 平移整数周期使其最接近 pred
        """
        diff = pred - target
        # 将 target 移动到离 pred 最近的周期副本
        shift = torch.round(diff)
        adjusted_target = target + shift
        
        return ((pred - adjusted_target) ** 2).mean()
    
    def forward(self, batch):
        if self.learning_stage == "flow":
            return self.flow(batch)
        else:
            return self.energy_matching(batch)

    def flow(self,batch,times = None,max_step = None):
        batch_size = batch.num_graphs
        if times == None:
            times = self.flow_beta_scheduler.uniform_sample_t(batch_size, self.device)
            max_step = self.flow_time_steps
        else:
            max_step = self.time_steps

        time_emb = self.time_embedding(times)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_x,rand_l = self.get_static_noise(frac_coords,lattices)
        input_lattice = rand_l+(lattices-rand_l)*times.view(-1,1,1)/max_step
        input_frac_coords = rand_x + self.wrapped_distance_vector(rand_x,frac_coords)*(times.repeat_interleave(batch.num_atoms)[:, None])/max_step

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        with torch.enable_grad():
            with RequiresGradContext(input_frac_coords, input_lattice, requires_grad=True):
                pred_e = (self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch))
                grad_f, grad_l = grad([pred_e.sum()], [input_frac_coords,input_lattice],create_graph=True,allow_unused=True)
        loss_lattice = F.mse_loss(-grad_l, lattices-rand_l)
        target_f = self.wrapped_distance_vector(rand_x,frac_coords)
        loss_coord = F.mse_loss((-grad_f), target_f)
        if self.i>100:
            if loss_coord>0.1:
                self._debug_flow_spike(
                    batch=batch,
                    times=times,
                    input_frac_coords=input_frac_coords,
                    input_lattice=input_lattice,
                    lattices=lattices,
                    rand_l=rand_l,
                    grad_f=grad_f,
                    grad_l=grad_l,
                    target_f=target_f,
                    loss_coord=loss_coord,
                )
        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_energy': torch.zeros_like(loss_coord)
        }

    def get_forces(self,frac_pos,cell,batch,time_emb):
        with torch.enable_grad():
            with RequiresGradContext(frac_pos, cell, requires_grad=True):
                pred_e = (self.decoder(time_emb, batch.atom_types, frac_pos,cell, batch.num_atoms, batch.batch))
                # grad_f, grad_l = grad(pred_e, [input_frac_coords,input_lattice], grad_outputs = grad_outputs,create_graph=True,allow_unused=True)
                grad_f, grad_l = grad([pred_e.sum()], [frac_pos,cell],create_graph=True,allow_unused=True)
                # grad_x = (grad_f.view(-1,1,3)@cell.detach().transpose(-1,-2).repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)
        return -grad_f,-grad_l,pred_e
    
    def langevin_step(self,pos,cell,batch,time_emb,step_size,std):
        tpos = pos.clone().detach()
        tcell = cell.clone().detach()
        fpos_forces,lattice_forces,_ = self.get_forces(tpos,tcell,batch,time_emb)
        pos = pos + step_size * fpos_forces + std*self.langevin_coord_noise * torch.randn_like(pos)
        cell=cell + step_size * lattice_forces + std *self.langevin_lattice_noise* torch.randn_like(cell)
        return pos,cell
    
    def energy_matching(self,batch):
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        flow_loss = self.flow(batch,times = times,max_step = self.time_steps)
        batch_size = batch.num_graphs
        # times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        # rand_times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        # times = torch.ones_like(rand_times)*rand_times.max()
        # time_emb = self.time_embedding(times)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_x,rand_l = self.get_static_noise(frac_coords,lattices,mode="log")#negative sampling
        # lattices,_ = self.rot_tril(lattices)

        l_pos = rand_x.clone().detach()
        l_cell = rand_l.clone().detach()
        for i in range(self.langevin_steps):
            now_time = self.dt*i
            times = torch.clamp(torch.ones_like(rand_times)*now_time*self.time_steps,min=0,max=self.time_steps)
            time_emb = self.time_embedding(times)
            l_pos,l_cell = self.langevin_step(l_pos.detach(),l_cell.detach(),batch,time_emb,step_size=self.dt,std=(2*self.dt*self.epsilon_strategy(eps=1.,step=i))**0.5)
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
        #for key in log_dict:
        #    print(key,":",log_dict[key],end=" |  ")
        #print("\n")
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

    @rank_zero_only
    def on_validation_epoch_end(self):
        self.i+=1
        if self.i>self.flow_warm_epochs:
            self.learning_stage="energy"
        m = self.trainer.callback_metrics

        msg = (
        f"[Epoch {self.current_epoch}] "
        f"val_loss={m['val_loss']:.4e}, "
        f"val_coord_loss={m['val_coord_loss']:.4e}, "
        f"val_lattice_loss={m['val_lattice_loss']:.4e}, "
        f"val_ene_loss={m['val_ene_loss']:.4e}"
        )
        print(msg)


class CSPEnergyMatching(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.flow_beta_scheduler = hydra.utils.instantiate(self.hparams.flow_beta_scheduler)
        self.lattice_noise_mode = self.hparams.lattice_noise_mode
        self.time_steps = self.hparams.timesteps
        self.flow_time_steps = self.hparams.flow_timesteps
        self.learning_stage = self.hparams.learning_stage
        self.dt = self.hparams.dt
        self.langevin_steps=self.hparams.langevin_steps
        self.lambda_cd = self.hparams.lambda_cd
        self.flow_warm_epochs = self.hparams.flow_warm_epochs
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.tau = self.hparams.tau
        self.langevin_coord_noise = float(getattr(self.hparams, "langevin_coord_noise", 0.1))
        self.langevin_lattice_noise = float(getattr(self.hparams, "langevin_lattice_noise", 0.5))
        self.sample_times = float(getattr(self.hparams, "langevin_sample_time", 3))
        self.flow_coord_huber_beta = float(getattr(self.hparams, "flow_coord_huber_beta", 0.4))
        self.use_ot = bool(getattr(self.hparams, "ot", False))
        # for test
        self.i = 0
        self._flow_spike_dump_count = 0

    @rank_zero_only
    def _dump_anomalous_force_structures(
        self,
        batch,
        times,
        input_frac_coords,
        input_lattice,
        pred_f,
        target_f,
        pred_l,
        target_l,
        loss_coord,
    ):
        pred_abs_threshold = float(getattr(self.hparams, "debug_pred_force_abs_threshold", 2.0))
        max_atoms_per_crystal = int(getattr(self.hparams, "debug_max_atoms_per_crystal", 24))
        max_crystals_per_batch = int(getattr(self.hparams, "debug_max_crystals_per_batch", 4))
        max_total_dumps = int(getattr(self.hparams, "debug_max_total_dumps", 2000))
        if self._flow_spike_dump_count >= max_total_dumps:
            return

        pred_f_detach = pred_f.detach()
        target_f_detach = target_f.detach()
        pred_l_detach = pred_l.detach()
        target_l_detach = target_l.detach()
        frac_detach = input_frac_coords.detach()
        lattice_detach = input_lattice.detach()
        atom_types_detach = batch.atom_types.detach()
        num_atoms_list = batch.num_atoms.detach().tolist()
        times_detach = times.detach()

        abs_max_component = pred_f_detach.abs().max(dim=-1).values
        abnormal_mask = abs_max_component > pred_abs_threshold
        abnormal_idx_global = torch.nonzero(abnormal_mask, as_tuple=False).flatten()
        if abnormal_idx_global.numel() == 0:
            return

        debug_root = Path(PROJECT_ROOT) / "debug_flow_spikes"
        debug_root.mkdir(parents=True, exist_ok=True)

        atom_start = 0
        dumped = 0
        for crystal_idx, n_atoms in enumerate(num_atoms_list):
            if dumped >= max_crystals_per_batch or self._flow_spike_dump_count >= max_total_dumps:
                break
            atom_end = atom_start + n_atoms
            crystal_abnormal_mask = abnormal_mask[atom_start:atom_end]
            if not crystal_abnormal_mask.any():
                atom_start = atom_end
                continue

            local_abnormal_idx = torch.nonzero(crystal_abnormal_mask, as_tuple=False).flatten()
            local_abs = abs_max_component[atom_start:atom_end][local_abnormal_idx]
            topk = min(max_atoms_per_crystal, local_abnormal_idx.numel())
            topk_local_order = torch.topk(local_abs, k=topk, largest=True).indices
            local_abnormal_idx = local_abnormal_idx[topk_local_order]

            crystal_frac = frac_detach[atom_start:atom_end] % 1.0
            crystal_cell = lattice_detach[crystal_idx]
            crystal_numbers = atom_types_detach[atom_start:atom_end].long().cpu().numpy()

            atoms = Atoms(
                numbers=crystal_numbers,
                scaled_positions=crystal_frac.cpu().numpy(),
                cell=crystal_cell.cpu().numpy(),
                pbc=True,
            )
            atoms.arrays["pred_f"] = pred_f_detach[atom_start:atom_end].cpu().numpy()
            atoms.arrays["target_f"] = target_f_detach[atom_start:atom_end].cpu().numpy()
            atoms.arrays["abnormal_force_atom"] = crystal_abnormal_mask.long().cpu().numpy()
            atoms.info["epoch"] = int(self.current_epoch)
            atoms.info["global_step"] = int(self.global_step)
            atoms.info["learning_stage"] = str(self.learning_stage)
            atoms.info["loss_coord"] = float(loss_coord.detach().item())
            atoms.info["interp_time"] = float(times_detach[crystal_idx].item())
            atoms.info["pred_lattice_force"] = pred_l_detach[crystal_idx].cpu().numpy().tolist()
            atoms.info["target_lattice_force"] = target_l_detach[crystal_idx].cpu().numpy().tolist()
            atoms.info["abnormal_atom_local_indices"] = local_abnormal_idx.cpu().numpy().tolist()

            file_stem = (
                f"epoch{int(self.current_epoch):04d}_step{int(self.global_step):08d}"
                f"_c{crystal_idx:03d}_t{int(times_detach[crystal_idx].item()):04d}"
            )
            out_path = debug_root / f"{file_stem}.extxyz"
            write(str(out_path), atoms, format="extxyz")

            detail_path = debug_root / f"{file_stem}.txt"
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write(f"loss_coord: {float(loss_coord.detach().item()):.8e}\n")
                f.write(f"interp_time: {float(times_detach[crystal_idx].item()):.6f}\n")
                f.write(f"crystal_idx: {crystal_idx}\n")
                f.write(f"num_atoms: {n_atoms}\n")
                f.write(f"pred_lattice_force:\n{pred_l_detach[crystal_idx].cpu().numpy()}\n")
                f.write(f"target_lattice_force:\n{target_l_detach[crystal_idx].cpu().numpy()}\n")
                f.write("abnormal_atoms(local_idx, pred_f, target_f):\n")
                for local_idx in local_abnormal_idx.cpu().tolist():
                    global_idx = atom_start + local_idx
                    f.write(
                        f"{local_idx:4d} "
                        f"pred={pred_f_detach[global_idx].cpu().numpy()} "
                        f"target={target_f_detach[global_idx].cpu().numpy()}\n"
                    )

            print(
                f"[flow-debug] dumped anomalous structure to {out_path} "
                f"(abnormal_atoms={local_abnormal_idx.numel()}, threshold={pred_abs_threshold:.2f})"
            )
            dumped += 1
            self._flow_spike_dump_count += 1
            atom_start = atom_end

        if dumped == 0:
            print(
                f"[flow-debug] no crystal-level dump, but found "
                f"{abnormal_idx_global.numel()} abnormal atoms globally"
            )

    def epsilon_strategy(self,eps,now_time):
        
        if now_time<self.tau:
            return 0.
        elif now_time>=self.tau and now_time<1.:
            return eps*(now_time - self.tau)/(1.-self.tau)
        else:
            return eps


    @staticmethod
    def wrapped_distance_vector(start_fpos,end_fpos):
        dis = end_fpos-start_fpos
        return dis-torch.round(dis)

    @staticmethod
    def _tensor_stats(name, tensor):
        if tensor is None:
            return f"{name}: None"
        flat = tensor.detach().reshape(-1).float()
        finite = torch.isfinite(flat)
        if finite.any():
            f = flat[finite]
            p95 = torch.quantile(f.abs(), 0.95)
            return (
                f"{name}: mean={f.mean().item():.4e}, "
                f"abs_mean={f.abs().mean().item():.4e}, "
                f"abs_max={f.abs().max().item():.4e}, "
                f"p95_abs={p95.item():.4e}, "
                f"norm={torch.linalg.norm(f).item():.4e}, "
                f"nan={torch.isnan(flat).sum().item()}, "
                f"inf={torch.isinf(flat).sum().item()}"
            )
        return (
            f"{name}: all_non_finite, "
            f"nan={torch.isnan(flat).sum().item()}, "
            f"inf={torch.isinf(flat).sum().item()}"
        )

    def _debug_flow_spike(
        self,
        batch,
        times,
        input_frac_coords,
        input_lattice,
        lattices,
        rand_l,
        grad_f,
        grad_l,
        target_f,
        loss_coord,
    ):
        num_atoms_f = batch.num_atoms.float()
        det_in = torch.det(input_lattice.detach())
        delta_l = (lattices - rand_l).detach()
        grad_ratio = torch.linalg.norm(grad_f.detach()) / (torch.linalg.norm(target_f.detach()) + 1e-8)
        pred_f = -grad_f
        pred_l = -grad_l
        target_l = delta_l

        print(f"[flow-debug] epoch={self.current_epoch}, stage={self.learning_stage}, loss_coord={loss_coord.item():.4e}")
        print(
            f"[flow-debug] times: min={times.min().item()}, "
            f"mean={times.float().mean().item():.2f}, max={times.max().item()}"
        )
        print(
            f"[flow-debug] num_atoms: min={num_atoms_f.min().item():.0f}, "
            f"mean={num_atoms_f.mean().item():.2f}, max={num_atoms_f.max().item():.0f}"
        )
        print(self._tensor_stats("target_f", target_f))
        print(self._tensor_stats("pred_f(-grad_f)", pred_f))
        print(self._tensor_stats("grad_l", grad_l))
        print(self._tensor_stats("det(input_lattice)", det_in))
        print(self._tensor_stats("delta_lattice(l-rand)", delta_l))
        print(f"[flow-debug] grad_f/target_f norm ratio={grad_ratio.item():.4e}")

        self._dump_anomalous_force_structures(
            batch=batch,
            times=times,
            input_frac_coords=input_frac_coords,
            input_lattice=input_lattice,
            pred_f=pred_f,
            target_f=target_f,
            pred_l=pred_l,
            target_l=target_l,
            loss_coord=loss_coord,
        )

    def get_static_noise(self,scaled_positions,cells,mode=None):
        scaled_positions_noise = torch.rand_like(scaled_positions)
        if mode == None:
            mode = self.lattice_noise_mode
        if mode == 2:
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
        elif mode == 3:
            loc,scale = 1.3595351865425995,0.1277552273430223
            cell_l = torch.exp(scale*torch.randn(size=(cells.shape[0],3),device=cells.device,dtype=cells.dtype)+loc)
            cell_a = 60+60*torch.rand(size=(cells.shape[0],3),device=cells.device,dtype=cells.dtype)
            cells_noise = self.lattice_from_parameters(cell_l[:,0],
                                                    cell_l[:,1],
                                                    cell_l[:,2],
                                                    cell_a[:,0],
                                                    cell_a[:,1],
                                                    cell_a[:,2])
        elif mode == 20:
            a_low,a_high = 2.4,12.8
            alpha_low,alpha_high = 60,120
            low = torch.tensor([a_low,a_low,a_low,alpha_low,alpha_low,alpha_low],device=cells.device,dtype=cells.dtype)
            high = torch.tensor([a_high,a_high,a_high,alpha_high,alpha_high,alpha_high],device=cells.device,dtype=cells.dtype)
            cells_noise_6d = low + (high - low) * torch.rand(size=(cells.shape[0],low.shape[0]),device=cells.device,dtype=cells.dtype)
            cells_length = cells_noise_6d[:,:3]
            cells_angle = cells_noise_6d[:,3:]
            cells_noise = lattice_params_to_matrix_torch(cells_length,cells_angle)
        
        elif mode == 1:
            cells_noise = torch.rand_like(cells)
        elif mode == 0:
            cells_noise = cells.detach().clone()

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

    @staticmethod
    def unwrapped_mse(pred, target):
        """
        pred:   模型预测的梯度, (B, N, 3), 值域 R^3
        target: 测地线速度,    (B, N, 3), 在 (-0.5, 0.5]
        
        将 target 平移整数周期使其最接近 pred
        """
        diff = pred - target
        # 将 target 移动到离 pred 最近的周期副本
        shift = torch.round(diff)
        adjusted_target = target + shift
        
        return ((pred - adjusted_target) ** 2).mean()
    
    def forward(self, batch):
        if self.learning_stage == "flow":
            return self.flow(batch)
        else:
            return self.energy_matching(batch)

    @staticmethod
    def _ot_match_noise_to_coords(rand_x, frac_coords, batch_idx, atom_types):
        """
        For each (crystal, element) group, reorder rand_x (noise) so that
        the assignment minimises Σ ||rand_x_σ(i) - frac_coords_i||².
        This ensures trajectories are as short as possible and do not cross,
        preventing large gradient spikes during flow training.
        """
        new_rand_x = rand_x.clone()
        for g in batch_idx.unique():
            crystal_mask = (batch_idx == g)
            for z in atom_types[crystal_mask].unique():
                group_mask = crystal_mask & (atom_types == z)
                idx = group_mask.nonzero(as_tuple=True)[0]
                if idx.numel() <= 1:
                    continue
                src = rand_x[idx]       # (n, 3) – noise
                tgt = frac_coords[idx]  # (n, 3) – target (fixed)
                # cost[i, j] = ||rand_x[i] - frac_coords[j]||² (wrapped)
                diff = tgt.unsqueeze(0) - src.unsqueeze(1)  # (n_src, n_tgt, 3)
                diff = diff - torch.round(diff)
                cost = (diff ** 2).sum(-1)                  # (n_src, n_tgt)
                # minimise Σ cost[σ(i), i]  →  solve on transposed cost
                # linear_sum_assignment returns (arange(n), col_perm); col_perm[i] = σ(i)
                _, col_ind = linear_sum_assignment(cost.T.detach().cpu().numpy())
                col_ind_t = torch.from_numpy(col_ind).to(idx.device)
                new_rand_x[idx] = src[col_ind_t]
        return new_rand_x

    def flow(self,batch,times = None,max_step = None):
        batch_size = batch.num_graphs
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords
        if times == None:
            times = self.flow_beta_scheduler.uniform_sample_t(batch_size, self.device)
            max_step = self.flow_time_steps
        else:
            max_step = self.time_steps
        rand_x,rand_l = self.get_static_noise(frac_coords,lattices)
        if self.use_ot:
            rand_x = self._ot_match_noise_to_coords(rand_x, frac_coords, batch.batch, batch.atom_types)
        input_frac_coords = rand_x + self.wrapped_distance_vector(rand_x,frac_coords)*(times.repeat_interleave(batch.num_atoms)[:, None])/max_step

        rand_l = extract_spd(rand_l)
        lattices = extract_spd(lattices)

        deform_1 = torch.bmm(rand_l.inverse(),lattices)
        log_deform_1 = matrix_log(deform_1)
        nan_mask = torch.isnan(log_deform_1).any(dim=-1).any(dim=-1)  # (B,)
        if nan_mask.any():
            for i in nan_mask.nonzero(as_tuple=True)[0]:
                print(f"[flow NaN] batch item {i.item()}")
                print(f"  rand_l[i]    =\n{rand_l[i].detach().cpu()}")
                print(f"  lattices[i]  =\n{lattices[i].detach().cpu()}")
                print(f"  deform_1[i]  =\n{deform_1[i].detach().cpu()}")
                print(f"  det(rand_l)  = {torch.linalg.det(rand_l[i]).item():.6e}")
                print(f"  det(lattices)= {torch.linalg.det(lattices[i]).item():.6e}")
                print(f"  inv(rand_l)= {(lattices[i].inverse()).item():.6e}")
        log_deform_t = times.view(-1,1,1)* log_deform_1/max_step
        deform_t = torch.matrix_exp(log_deform_t)
        input_lattice = torch.bmm(rand_l,deform_t)
        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
            deform_t = torch.eye(3, device=lattices.device).unsqueeze(0).repeat(batch_size,1,1)

        

        input_cart_coords = (input_frac_coords.reshape(-1,1,3)@input_lattice.repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)
        with torch.enable_grad():
            with RequiresGradContext(input_cart_coords, log_deform_t, requires_grad=True):
                deform_t = torch.matrix_exp(log_deform_t)
                context_input_lattice = torch.bmm(rand_l, deform_t)
                context_input_frac_coords = (input_cart_coords.reshape(-1,1,3)@context_input_lattice.inverse().repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)
                pred_e = self.decoder(batch.atom_types, context_input_frac_coords, context_input_lattice, batch.num_atoms, batch.batch)
                grad_x, grad_d = grad([pred_e.sum()], [input_cart_coords,log_deform_t],create_graph=True,allow_unused=True)

        velocity_f = (-grad_x.reshape(-1,1,3)@input_lattice.inverse().repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)        
        loss_coord = F.mse_loss(velocity_f, self.wrapped_distance_vector(rand_x,frac_coords))

        deform_diff = log_deform_1
        loss_lattice = F.mse_loss(-grad_d/torch.det(input_lattice.detach()).view(-1,1,1), deform_diff)
        if self.i>100:
            if loss_coord>0.1:
                self._debug_flow_spike(
                    batch=batch,
                    times=times,
                    input_frac_coords=input_frac_coords,
                    input_lattice=input_lattice,
                    lattices=lattices,
                    rand_l=rand_l,
                    grad_f=velocity_f,
                    grad_l=grad_d,
                    target_f=self.wrapped_distance_vector(rand_x,frac_coords),
                    loss_coord=loss_coord,
                )
        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_energy': torch.zeros_like(loss_coord)
        }

    def get_forces(self,frac_pos,cell,batch,decoder = None):
        if decoder == None:
            decoder = self.decoder
        with torch.enable_grad():
            with RequiresGradContext(frac_pos, cell, requires_grad=True):
                pred_e = (decoder(batch.atom_types, frac_pos,cell, batch.num_atoms, batch.batch))
                # grad_f, grad_l = grad(pred_e, [input_frac_coords,input_lattice], grad_outputs = grad_outputs,create_graph=True,allow_unused=True)
                grad_f, grad_l = grad([pred_e.sum()], [frac_pos,cell],create_graph=True,allow_unused=True)
                # grad_x = (grad_f.view(-1,1,3)@cell.detach().transpose(-1,-2).repeat_interleave(batch.num_atoms,dim=0)).squeeze(1)
        return -grad_f,-grad_l,pred_e
    
    def langevin_step(self,pos,cell,batch,step_size,std,decoder = None):
        tpos = pos.clone().detach()
        tcell = cell.clone().detach()
        fpos_forces,lattice_forces,_ = self.get_forces(tpos,tcell,batch,decoder=decoder)
        pos = pos + step_size * fpos_forces + std*self.langevin_coord_noise * torch.randn_like(pos)
        cell=cell + step_size * lattice_forces + std *self.langevin_lattice_noise* torch.randn_like(cell)
        return pos,cell
    
    def energy_matching(self,batch):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        flow_loss = self.flow(batch,times = times,max_step = self.time_steps)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_x,rand_l = self.get_static_noise(frac_coords,lattices)#negative sampling

        l_pos = rand_x.clone().detach()
        l_cell = rand_l.clone().detach()
        for i in range(self.langevin_steps):
            now_time = self.dt*i
            l_pos,l_cell = self.langevin_step(l_pos.detach(),l_cell.detach(),batch,step_size=self.dt,std=(2*self.dt*self.epsilon_strategy(eps=1.,now_time=now_time))**0.5)
            l_pos = wrap_coordinates(l_pos)
        _,_,neg_e = self.get_forces(l_pos,l_cell,batch)
        _,_,pos_e = self.get_forces(frac_coords,lattices,batch)
        energy_loss = torch.mean(pos_e)-torch.mean(neg_e)
        flow_loss["loss_energy"] = self.lambda_cd*energy_loss
        flow_loss["loss"]+=energy_loss*self.lambda_cd
        return flow_loss
    

    @torch.no_grad()
    def sample(self, batch,sample_time = -1,eps=1.0):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        scaled_positions_noise,cells_noise = self.get_static_noise(scaled_positions=x_T,cells=l_T)
        if sample_time <1:
            sample_time = self.sample_times
        self.sample_langevin_steps = int(sample_time/self.dt)

        l_fpos = scaled_positions_noise.clone().detach()
        l_cell = cells_noise.clone().detach()

        traj = {}
        langevin_step = 0
        for langevin_step in tqdm(range(0,self.sample_langevin_steps,1)):
            traj[langevin_step] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : l_fpos % 1.,
                'lattices' : l_cell              
            }
            now_time = self.dt * langevin_step
            l_fpos,l_cell = self.langevin_step(l_fpos.detach(),l_cell.detach(),batch,
                               step_size=self.dt,std=(2*self.dt*self.epsilon_strategy(eps=eps,now_time=now_time))**0.5,
                               decoder=self.decoder)

        traj[langevin_step+1] ={
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : l_fpos % 1.,
                'lattices' : l_cell              
        }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(0, langevin_step+2, 1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(0, langevin_step+2, 1)])
        }

        res = traj[langevin_step+1]
        res['atom_types'] = batch.atom_types

        return traj[langevin_step+1], traj_stack


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
            self.zero_grad()
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')
        #for key in log_dict:
        #    print(key,":",log_dict[key],end=" |  ")
        #print("\n")
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

    @rank_zero_only
    def on_validation_epoch_end(self):
        self.i+=1
        if self.i==self.flow_warm_epochs:
            self.learning_stage="energy"
            print("[Energy-Matching Log]: Now switch to energy-matching strategy.")
        m = self.trainer.callback_metrics

        msg = (
        f"[Epoch {self.current_epoch}] "
        f"val_loss={m['val_loss']:.4e}, "
        f"val_coord_loss={m['val_coord_loss']:.4e}, "
        f"val_lattice_loss={m['val_lattice_loss']:.4e}, "
        f"val_ene_loss={m['val_ene_loss']:.4e}"
        )
        print(msg)
