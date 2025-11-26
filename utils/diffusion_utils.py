import torch
import torch.nn as nn
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from Improved DDPM paper
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    """
    Linear schedule from original DDPM
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionProcess:
    """
    Implements forward and reverse diffusion processes
    """
    def __init__(self, num_timesteps=1000, schedule='cosine'):
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Store for q_sample
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - alphas_cumprod))
        
        # Store for p_sample
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', 
                            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        """Helper to store tensors as attributes"""
        setattr(self, name, tensor)
    
    def extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to ground-truth boxes
        x_start: (B, N, 4) ground truth boxes
        t: (B,) timestep
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x, t, img_features):
        """
        Single step of reverse diffusion
        """
        # Get model prediction
        pred_noise = model(x, t, img_features)
        
        # Compute x_{t-1}
        beta_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, model, shape, img_features, device):
        """
        Complete reverse process: generate predictions from noise
        """
        b = shape[0]
        # Start from pure noise
        boxes = torch.randn(shape, device=device)
        
        boxes_trajectory = []
        
        # Iterate backwards from T to 0
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            boxes = self.p_sample(model, boxes, t, img_features)
            boxes_trajectory.append(boxes.clone())
        
        return boxes, boxes_trajectory