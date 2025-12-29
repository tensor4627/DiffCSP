import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..utils import wrap, get_batch_data, get_grad_of_wnd,CellFilter,get_uni_batch_data,polar_decompose_right
import logging
from scipy.linalg import logm

log = logging.getLogger(__name__)


class CrystalDiffusion(nn.Module):
    def __init__(self, cutoff, t_max, betas, sigmas) -> None:
        super().__init__()
        betas = np.array(betas)
        sigmas = np.array(sigmas)
        self.register_buffer("t_max", torch.tensor(t_max, dtype=torch.long))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        # noise parameters for cell diffusion
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer(
            "alpha_bars", torch.tensor(alpha_bars, dtype=torch.float32)
        )
        # noise parameters for fraction positions
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))

    def get_diffusion(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must have 'get_diffusion'!"
        )

    def forward(self, batch_data):
        n_atoms = batch_data["n_atoms"]
        atomic_numbers = batch_data["atomic_numbers"]
        cells = batch_data["cells"]
        limit_cells = batch_data["limit_cells"]
        scaled_positions = batch_data["scaled_positions"]
        t = torch.randint_like(n_atoms, 0, self.t_max)  # [n_batch]

        # add noise to fraction positions
        # sample noise levels.
        sigma_t = (self.sigmas[t] / n_atoms ** (1 / 3)).repeat_interleave(
            n_atoms
        )  # [n_atoms]
        # add noise to the cart coords
        scaled_positions_noise = (
            torch.randn_like(scaled_positions) * sigma_t[:, None]
        )  # [natoms, 3]
        # !!!!!!!!!!!!!!!
        # scaled_positions_noise = torch.zeros_like(scaled_positions)

        scaled_positions_t = wrap(scaled_positions + scaled_positions_noise)

        # add noise to cell
        # sample noise levels.
        alpha_bar_t = self.alpha_bars[t][:, None, None]  # [n_batch, 1, 1]
        # add noise to the cell
        cells_strain = torch.randn_like(cells)  # [n_batch, 3, 3]
        cells_strain[:, [1, 2, 2], [0, 0, 1]] = cells_strain[:, [0, 0, 1], [1, 2, 2]]
        cells_noise = cells @ cells_strain
        cells_t = (
            # torch.sqrt(alpha_bar_t) * cells
            cells
            # + (1 - torch.sqrt(alpha_bar_t)) * limit_cells
            + torch.sqrt(1 - alpha_bar_t) * cells_noise  # * 0.1
        )
        # predict epsilon_l and epsilon_f
        batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            scaled_positions_t.detach().clone().cpu().numpy(),
            cells_t.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        epsilon_f, predict_stress = self.get_diffusion(batch_data, t)
        frac_loss = F.mse_loss(
            self.get_frac_score(scaled_positions_noise, sigma_t), epsilon_f
        )
        stress = torch.linalg.inv(cells_t) @ cells
        predict_noise = torch.linalg.inv(predict_stress)
        predict_noise[:, [0, 1, 2], [0, 1, 2]] -= 1
        predict_noise = predict_noise / torch.sqrt(1 - alpha_bar_t)
        cell_loss = F.mse_loss(predict_stress, stress)
        hehe = torch.sqrt(F.mse_loss(predict_noise, cells_strain))
        xixi = torch.sqrt(F.mse_loss(cells_strain, torch.zeros_like(stress)))
        print(hehe.item(), xixi.item())
        print(np.round(predict_stress[0].detach().cpu().numpy(), 3))
        print(np.round(stress[0].detach().cpu().numpy(), 3))
        print("==========================================")
        print(np.round(predict_noise[0].detach().cpu().numpy(), 3))
        print(np.round(cells_strain[0].detach().cpu().numpy(), 3))
        return frac_loss, cell_loss

    @torch.no_grad()
    def get_frac_score(self, scaled_positions_noise, sigma_t, k_max=10):
        score = get_grad_of_wnd(
            scaled_positions_noise.reshape(-1), sigma_t.repeat_interleave(3), k_max
        )
        return score.reshape(-1, 3) * sigma_t[:, None] ** 2

class DiffCSPEM(nn.Module):
    def __init__(self, cutoff, t_max, betas, sigmas) -> None:
        super().__init__()
        betas = np.array(betas)
        sigmas = np.array(sigmas)
        self.register_buffer("t_max", torch.tensor(t_max, dtype=torch.long))
        self.register_buffer("t_start", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        # noise parameters for cell diffusion
        self.cellfilter = CellFilter()

    def model_predict(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must have 'model_predict'!"
        )

    def CD_loss(self,Vdata,Vneg):
        return torch.mean(Vdata,dim=-1)-torch.mean(Vneg,dim=-1)
    
    def frac_epsilon_strategy(self,t):
        pass
    
    def cell_epsilon_strategy(self,t):
        pass
    
    def uni_langevin_step(self,uni_pos,uni_forces,delta_t,t):
        eps = self.frac_epsilon_strategy(t)
        eta = torch.randn_like(new_uni_pos)
        new_uni_pos = uni_pos+delta_t*uni_forces+(2*delta_t*eps)**(0.5)*eta
        return new_uni_pos

    def sample_negative(self,batch_data):
        return self.get_noise(batch_data,self.t_max)
    
    def iter_langevin(self,batch_data):
        n_atoms =batch_data["n_atoms"]
        cells = batch_data["cells"]
        atomic_numbers = batch_data["atomic_numbers"]
        for time in range(0,self.t_max):
            t = torch.ones_like(n_atoms)*time
            self.model_predict(batch_data,t)
            forces_p = batch_data["forces_p"]#[natoms,3]
            virial_p = batch_data["virial_p"]#[nbatch,3,3]
            uni_pos_t,uni_force_t = self.cellfilter.transform_fall(pos_neg,cell_neg,cells,forces_p,virial_p,n_atoms)
            new_uni_pos_t = self.uni_langevin_step(uni_pos_t,uni_force_t,self.delta_t,t)
            pos_neg,cell_neg = self.cellfilter.rebuild_fall(new_uni_pos_t,cells,n_atoms)

            batch_data = get_batch_data(
                n_atoms.detach().clone().cpu().numpy(),
                atomic_numbers.detach().clone().cpu().numpy(),
                pos_neg.detach().clone().cpu().numpy(),
                cell_neg.detach().clone().cpu().numpy(),
                float(self.cutoff.detach().clone().cpu().numpy()),
                device=n_atoms.device,
            )
        self.model_predict(batch_data,t)
        return pos_neg,cell_neg,batch_data["energy_p"]

    
    def get_noise(self,batch_data):
        n_atoms = batch_data["n_atoms"]#[nbatch]
        cells = batch_data["cells"]#[nbatch,3,3]
        cell_length = torch.linalg.norm(cells,dim=2)
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
        scaled_positions = batch_data["scaled_positions"]#[natoms,3]
        scaled_positions_noise = torch.rand_like(scaled_positions)
        return scaled_positions_noise,cells_noise

    @staticmethod
    def lattice_from_parameters(a, b, c, alpha, beta, gamma, degrees=True):
        """
        a, b, c: (...,)  张量
        alpha, beta, gamma: (...,) 张量
        return: (..., 3, 3) 晶格矩阵
        """
        if degrees:
            alpha = torch.deg2rad(alpha)
            beta  = torch.deg2rad(beta)
            gamma = torch.deg2rad(gamma)

        # 计算三角函数
        cos_alpha = torch.cos(alpha)
        cos_beta  = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        # a1
        a1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)

        # a2
        a2 = torch.stack([
            b * cos_gamma,
            b * sin_gamma,
            torch.zeros_like(b)
        ], dim=-1)

        # a3
        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = c * torch.sqrt(
            1 - cos_beta**2
            - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2
        )

        a3 = torch.stack([cx, cy, cz], dim=-1)

        # 拼成矩阵 (..., 3, 3)
        lattice = torch.stack([a1, a2, a3], dim=-2)
        return lattice

    @staticmethod
    def _exp_interpolation(cells,cells_noise,t):
        pass

    @staticmethod
    def _unit_interpolation(cells,cells_noise,t,t_max):
        cells_t = cells_noise+t.view(-1,1,1)*(cells-cells_noise)/t_max


    

    def flow(self,batch_data):
        n_atoms = batch_data["n_atoms"]
        atomic_numbers = batch_data["atomic_numbers"]
        cells = batch_data["cells"]
        scaled_positions = batch_data["scaled_positions"]
        t = torch.randint_like(n_atoms, self.t_start, self.t_max+1)  # [n_batch] for warm up only
        # get_noise
        # scaled_positions_noise = torch.rand_like(scaled_positions)
        scaled_positions_noise,cells_noise = self.get_noise(batch_data)
        _,cells = polar_decompose_right(cells)
        _,cells_noise = polar_decompose_right(cells_noise)

        # time_interpolation
        wrapped_frac_distance = (scaled_positions-scaled_positions_noise-torch.round(scaled_positions-scaled_positions_noise))
        deform = torch.bmm(torch.inverse(cells_noise),cells)
        frac_t = wrap(scaled_positions_noise+t.repeat_interleave(n_atoms)[:, None]*(wrapped_frac_distance)/self.t_max)
        deform_0 = torch.eye(3,dtype=deform.dtype,device=deform.device).reshape(-1,3,3).expand(n_atoms.shape[0],3,3)
        deform_t = deform_0+(deform-deform_0)*t.view(-1,1,1)/self.t_max
        cells_t = torch.bmm(cells_noise,deform_t)
        
        #predict_energy
        batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            frac_t.detach().clone().cpu().numpy(),
            cells_t.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        self.model_predict(batch_data,t)
        frac_velocity = batch_data["frac_velocity"]#[natoms,3]
        defrom_velocity = batch_data["deform"]
        frac_loss = F.mse_loss(frac_velocity,wrapped_frac_distance)
        cell_loss = F.mse_loss(defrom_velocity,deform)
        CD_loss = torch.zeros_like(cell_loss)
        return frac_loss,cell_loss,CD_loss


    def energy_matching(self,batch_data):
        n_atoms = batch_data["n_atoms"]
        atomic_numbers = batch_data["atomic_numbers"]
        cells = batch_data["cells"]
        limit_cells = batch_data["limit_cells"]

        scaled_positions = batch_data["scaled_positions"]

        OT_frac_loss,OT_cell_loss = self.flow(batch_data,t_start=0)


        t = torch.randint_like(n_atoms, 0, self.t_max)  # [n_batch] for warm up only

        #add noise
        sigma_t = (self.sigmas[t] / n_atoms ** (1 / 3)).repeat_interleave(
            n_atoms
        )  # [n_atoms]
        # add noise to the cart coords
        scaled_positions_noise = (
            torch.randn_like(scaled_positions) * sigma_t[:, None]
        )  # [natoms, 3]

        cell_noise = torch.randn_like(cells)
        frac_t = wrap((1-t/self.t_max)*scaled_positions_noise+t/self.t_max*scaled_positions) # did it need wrap here?
        cells_t = (1-t/self.t_max)*cell_noise+t/self.t_max*cells

        flow_batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            frac_t.detach().clone().cpu().numpy(),
            cells_t.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        pos_batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            scaled_positions.detach().clone().cpu().numpy(),
            cells.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        pos_neg,cell_neg = self.sample_negative(batch_data)
        neg_batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            pos_neg.detach().clone().cpu().numpy(),
            cell_neg.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )

        OT_frac_loss,OT_cell_loss, OT_CD_loss = self.flow(flow_batch_data)
        self.model_predict(pos_batch_data,t=torch.zeros_like(n_atoms)*self.t_max)
        E_pos = pos_batch_data["energy_p"]
        _,_,E_neg = self.iter_langevin(neg_batch_data)
        ebm_loss = E_pos-E_neg
        return OT_frac_loss,OT_cell_loss,ebm_loss
    


class CrystalEM(nn.Module):
    def __init__(self, cutoff, t_max, betas, sigmas) -> None:
        super().__init__()
        betas = np.array(betas)
        sigmas = np.array(sigmas)
        self.register_buffer("t_max", torch.tensor(t_max, dtype=torch.long))
        self.register_buffer("t_start", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        # noise parameters for cell diffusion
        self.cellfilter = CellFilter()

    def model_predict(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must have 'model_predict'!"
        )

    def CD_loss(self,Vdata,Vneg):
        return torch.mean(Vdata,dim=-1)-torch.mean(Vneg,dim=-1)
    
    def frac_epsilon_strategy(self,t):
        pass
    
    def cell_epsilon_strategy(self,t):
        pass
    
    def uni_langevin_step(self,uni_pos,uni_forces,delta_t,t):
        eps = self.frac_epsilon_strategy(t)
        eta = torch.randn_like(new_uni_pos)
        new_uni_pos = uni_pos+delta_t*uni_forces+(2*delta_t*eps)**(0.5)*eta
        return new_uni_pos

    def sample_negative(self,batch_data):
        return self.get_noise(batch_data,self.t_max)
    
    def iter_langevin(self,batch_data):
        n_atoms =batch_data["n_atoms"]
        cells = batch_data["cells"]
        atomic_numbers = batch_data["atomic_numbers"]
        for time in range(0,self.t_max):
            t = torch.ones_like(n_atoms)*time
            self.model_predict(batch_data,t)
            forces_p = batch_data["forces_p"]#[natoms,3]
            virial_p = batch_data["virial_p"]#[nbatch,3,3]
            uni_pos_t,uni_force_t = self.cellfilter.transform_fall(pos_neg,cell_neg,cells,forces_p,virial_p,n_atoms)
            new_uni_pos_t = self.uni_langevin_step(uni_pos_t,uni_force_t,self.delta_t,t)
            pos_neg,cell_neg = self.cellfilter.rebuild_fall(new_uni_pos_t,cells,n_atoms)

            batch_data = get_batch_data(
                n_atoms.detach().clone().cpu().numpy(),
                atomic_numbers.detach().clone().cpu().numpy(),
                pos_neg.detach().clone().cpu().numpy(),
                cell_neg.detach().clone().cpu().numpy(),
                float(self.cutoff.detach().clone().cpu().numpy()),
                device=n_atoms.device,
            )
        self.model_predict(batch_data,t)
        return pos_neg,cell_neg,batch_data["energy_p"]

    
    def get_noise(self,batch_data):
        n_atoms = batch_data["n_atoms"]#[nbatch]
        cells = batch_data["cells"]#[nbatch,3,3]
        cell_length = torch.linalg.norm(cells,dim=2)
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
        scaled_positions = batch_data["scaled_positions"]#[natoms,3]
        scaled_positions_noise = torch.rand_like(scaled_positions)
        return scaled_positions_noise,cells_noise

    @staticmethod
    def lattice_from_parameters(a, b, c, alpha, beta, gamma, degrees=True):
        """
        a, b, c: (...,)  张量
        alpha, beta, gamma: (...,) 张量
        return: (..., 3, 3) 晶格矩阵
        """
        if degrees:
            alpha = torch.deg2rad(alpha)
            beta  = torch.deg2rad(beta)
            gamma = torch.deg2rad(gamma)

        # 计算三角函数
        cos_alpha = torch.cos(alpha)
        cos_beta  = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        # a1
        a1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)

        # a2
        a2 = torch.stack([
            b * cos_gamma,
            b * sin_gamma,
            torch.zeros_like(b)
        ], dim=-1)

        # a3
        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = c * torch.sqrt(
            1 - cos_beta**2
            - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2
        )

        a3 = torch.stack([cx, cy, cz], dim=-1)

        # 拼成矩阵 (..., 3, 3)
        lattice = torch.stack([a1, a2, a3], dim=-2)
        return lattice

    @staticmethod
    def _exp_interpolation(cells,cells_noise,t):
        pass

    @staticmethod
    def _unit_interpolation(cells,cells_noise,t,t_max):
        cells_t = cells_noise+t.view(-1,1,1)*(cells-cells_noise)/t_max


    

    def flow(self,batch_data):
        n_atoms = batch_data["n_atoms"]
        atomic_numbers = batch_data["atomic_numbers"]
        cells = batch_data["cells"]
        scaled_positions = batch_data["scaled_positions"]
        t = torch.randint_like(n_atoms, self.t_start, self.t_max+1)  # [n_batch] for warm up only
        # get_noise
        # scaled_positions_noise = torch.rand_like(scaled_positions)
        scaled_positions_noise,cells_noise = self.get_noise(batch_data)
        _,cells = polar_decompose_right(cells)
        _,cells_noise = polar_decompose_right(cells_noise)

        # time_interpolation
        wrapped_frac_distance = (scaled_positions-scaled_positions_noise-torch.round(scaled_positions-scaled_positions_noise))
        deform = torch.bmm(torch.inverse(cells_noise),cells)
        frac_t = wrap(scaled_positions_noise+t.repeat_interleave(n_atoms)[:, None]*(wrapped_frac_distance)/self.t_max)
        deform_0 = torch.eye(3,dtype=deform.dtype,device=deform.device).reshape(-1,3,3).expand(n_atoms.shape[0],3,3)
        deform_t = deform_0+(deform-deform_0)*t.view(-1,1,1)/self.t_max
        cells_t = torch.bmm(cells_noise,deform_t)
        
        #predict_energy
        batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            frac_t.detach().clone().cpu().numpy(),
            cells_t.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        self.model_predict(batch_data,t)
        frac_velocity = batch_data["frac_velocity"]#[natoms,3]
        defrom_velocity = batch_data["deform"]
        frac_loss = F.mse_loss(frac_velocity,wrapped_frac_distance)
        cell_loss = F.mse_loss(defrom_velocity,deform)
        CD_loss = torch.zeros_like(cell_loss)
        return frac_loss,cell_loss,CD_loss


    def energy_matching(self,batch_data):
        n_atoms = batch_data["n_atoms"]
        atomic_numbers = batch_data["atomic_numbers"]
        cells = batch_data["cells"]
        limit_cells = batch_data["limit_cells"]

        scaled_positions = batch_data["scaled_positions"]

        OT_frac_loss,OT_cell_loss = self.flow(batch_data,t_start=0)


        t = torch.randint_like(n_atoms, 0, self.t_max)  # [n_batch] for warm up only

        #add noise
        sigma_t = (self.sigmas[t] / n_atoms ** (1 / 3)).repeat_interleave(
            n_atoms
        )  # [n_atoms]
        # add noise to the cart coords
        scaled_positions_noise = (
            torch.randn_like(scaled_positions) * sigma_t[:, None]
        )  # [natoms, 3]

        cell_noise = torch.randn_like(cells)
        frac_t = wrap((1-t/self.t_max)*scaled_positions_noise+t/self.t_max*scaled_positions) # did it need wrap here?
        cells_t = (1-t/self.t_max)*cell_noise+t/self.t_max*cells

        flow_batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            frac_t.detach().clone().cpu().numpy(),
            cells_t.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        pos_batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            scaled_positions.detach().clone().cpu().numpy(),
            cells.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )
        pos_neg,cell_neg = self.sample_negative(batch_data)
        neg_batch_data = get_batch_data(
            n_atoms.detach().clone().cpu().numpy(),
            atomic_numbers.detach().clone().cpu().numpy(),
            pos_neg.detach().clone().cpu().numpy(),
            cell_neg.detach().clone().cpu().numpy(),
            float(self.cutoff.detach().clone().cpu().numpy()),
            device=n_atoms.device,
        )

        OT_frac_loss,OT_cell_loss, OT_CD_loss = self.flow(flow_batch_data)
        self.model_predict(pos_batch_data,t=torch.zeros_like(n_atoms)*self.t_max)
        E_pos = pos_batch_data["energy_p"]
        _,_,E_neg = self.iter_langevin(neg_batch_data)
        ebm_loss = E_pos-E_neg
        return OT_frac_loss,OT_cell_loss,ebm_loss