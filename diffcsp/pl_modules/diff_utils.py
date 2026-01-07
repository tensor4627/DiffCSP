import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal(x, sigma, N, T)

def sigma_norm(sigma, T=1.0, sn = 10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T = T)
    return (normal_ ** 2).mean(dim = 0)




class BetaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        scheduler_mode,
        beta_start = 0.0001,
        beta_end = 0.02
    ):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)


        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)

        sigmas[1:] = betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])

        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sigmas', sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)

class SigmaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        sigma_begin = 0.01,
        sigma_end = 1.0
    ):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)))

        sigmas_norm_ = sigma_norm(sigmas)

        self.register_buffer('sigmas', torch.cat([torch.zeros([1]), sigmas], dim=0))
        self.register_buffer('sigmas_norm', torch.cat([torch.ones([1]), sigmas_norm_], dim=0))

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)
    



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

def get_static_noise(scaled_positions,cells):
    scaled_positions_noise = torch.rand_like(scaled_positions)
    a_low,a_high = 2.4,12.8
    alpha_low,alpha_high = 60,120
    low = torch.tensor([a_low,a_low,a_low,alpha_low,alpha_low,alpha_low],device=cells.device,dtype=cells.dtype)
    high = torch.tensor([a_high,a_high,a_high,alpha_high,alpha_high,alpha_high],device=cells.device,dtype=cells.dtype)
    cells_noise_6d = low + (high - low) * torch.rand(size=(cells.shape[0],low.shape[0]),device=cells.device,dtype=cells.dtype)
    cells_noise = lattice_from_parameters(cells_noise_6d[:,0],
                                                cells_noise_6d[:,1],
                                                cells_noise_6d[:,2],
                                                cells_noise_6d[:,3],
                                                cells_noise_6d[:,4],
                                                cells_noise_6d[:,5])
    return scaled_positions_noise,cells_noise       


def rot_tril(H):
    """
    Rotate lattice matrices to a lower-triangular gauge with positive diagonal.

    Parameters
    ----------
    H : torch.Tensor
        Shape (B, 3, 3). Columns are lattice vectors.

    Returns
    -------
    L : torch.Tensor
        Shape (B, 3, 3). Lower-triangular lattice with positive diagonal.
    R_rot : torch.Tensor
        Shape (B, 3, 3). Rotation matrix in SO(3), satisfying R_rot @ H = L.
    """
    # Step 1: QR decomposition of H^T
    # H^T = Q R
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
