"""
Latent space sampling utilities for generative music models.
Supports sampling from Autoencoder, VAE, and interpolation experiments.
"""
import torch
import numpy as np

from src.config import AE_LATENT_DIM, VAE_LATENT_DIM, DEVICE


def sample_random_latent(latent_dim: int, num_samples: int = 1,
                         device: torch.device = DEVICE) -> torch.Tensor:
    """Sample random latent vectors from N(0, I)."""
    return torch.randn(num_samples, latent_dim, device=device)


def sample_interpolated_latent(z1: torch.Tensor, z2: torch.Tensor,
                               num_steps: int = 10) -> list[torch.Tensor]:
    """Linear interpolation between two latent vectors.

    Args:
        z1: (1, latent_dim) start point
        z2: (1, latent_dim) end point
        num_steps: Number of interpolation steps

    Returns:
        List of interpolated latent vectors
    """
    alphas = torch.linspace(0, 1, num_steps)
    return [(1 - alpha) * z1 + alpha * z2 for alpha in alphas]


def sample_spherical_interpolation(z1: torch.Tensor, z2: torch.Tensor,
                                   num_steps: int = 10) -> list[torch.Tensor]:
    """Spherical linear interpolation (SLERP) between two latent vectors."""
    z1_norm = z1 / z1.norm(dim=-1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=-1, keepdim=True)

    cos_angle = (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
    angle = torch.acos(cos_angle)

    results = []
    for alpha in torch.linspace(0, 1, num_steps):
        if angle.abs() < 1e-6:
            z = (1 - alpha) * z1 + alpha * z2
        else:
            z = (torch.sin((1 - alpha) * angle) * z1 + torch.sin(alpha * angle) * z2) / torch.sin(angle)
        results.append(z)
    return results


def sample_around_point(z_center: torch.Tensor, radius: float = 0.5,
                        num_samples: int = 10) -> list[torch.Tensor]:
    """Sample points in a neighborhood around a center point in latent space."""
    device = z_center.device
    latent_dim = z_center.shape[-1]
    perturbations = torch.randn(num_samples, latent_dim, device=device) * radius
    return [z_center + p.unsqueeze(0) for p in perturbations]


def encode_and_sample(model, x: torch.Tensor, num_variations: int = 5,
                      radius: float = 0.3, model_type: str = "ae") -> list[torch.Tensor]:
    """Encode an input piece and generate variations around it.

    Args:
        model: Trained autoencoder or VAE model
        x: (1, seq_len, input_dim) input sequence
        num_variations: Number of variations to generate
        radius: Perturbation radius in latent space
        model_type: "ae" for autoencoder, "vae" for VAE

    Returns:
        List of generated piano-roll tensors
    """
    model.eval()
    with torch.no_grad():
        if model_type == "vae":
            mu, logvar = model.encoder(x)
            z_center = mu
        else:
            z_center = model.encode(x)

        samples = sample_around_point(z_center, radius, num_variations)
        outputs = []
        for z in samples:
            x_hat = model.decode(z)
            outputs.append(x_hat)
        return outputs
