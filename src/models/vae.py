"""
Task 2: Variational Autoencoder (VAE) for Multi-Genre Music Generation.

Architecture:
    Encoder: LSTM that outputs mean μ(X) and log-variance σ(X)
    Sampling: z = μ + σ ⊙ ε,  ε ~ N(0, I)  (reparameterization trick)
    Decoder: LSTM that reconstructs X_hat from z

Loss: L_VAE = L_recon + β · D_KL(q_φ(z|X) || p(z))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    NUM_PITCHES, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_NUM_LAYERS,
    VAE_DROPOUT, VAE_BETA, SEQUENCE_LENGTH, NUM_GENRES, DEVICE,
)


class VAEEncoder(nn.Module):
    """LSTM Encoder for VAE — outputs μ and log(σ²)."""

    def __init__(self, input_dim: int = NUM_PITCHES, hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM, num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        mu = self.fc_mu(h_cat)
        logvar = self.fc_logvar(h_cat)
        return mu, logvar


class VAEDecoder(nn.Module):
    """LSTM Decoder for VAE — reconstructs sequence from z (optionally conditioned on genre)."""

    def __init__(self, latent_dim: int = VAE_LATENT_DIM, hidden_dim: int = VAE_HIDDEN_DIM,
                 output_dim: int = NUM_PITCHES, num_layers: int = VAE_NUM_LAYERS,
                 seq_len: int = SEQUENCE_LENGTH, dropout: float = VAE_DROPOUT,
                 num_genres: int = NUM_GENRES):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Genre embedding (optional conditioning)
        self.genre_embedding = nn.Embedding(num_genres, latent_dim)
        decoder_input_dim = latent_dim  # genre embedding is added to z

        self.fc_init = nn.Linear(decoder_input_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(decoder_input_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, genre: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
            genre: (batch,) optional genre indices
        Returns:
            x_hat: (batch, seq_len, output_dim)
        """
        if genre is not None:
            genre_emb = self.genre_embedding(genre)  # (batch, latent_dim)
            z = z + genre_emb

        batch_size = z.size(0)
        h_0 = self.fc_init(z).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = self.fc_cell(z).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        decoder_input = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        output, _ = self.lstm(decoder_input, (h_0, c_0))
        x_hat = torch.sigmoid(self.fc_out(output))
        return x_hat


class MusicVAE(nn.Module):
    """Variational Autoencoder for Multi-Genre Music Generation (Task 2).

    q_φ(z|X) = N(μ(X), σ(X))
    z = μ + σ ⊙ ε,    ε ~ N(0, I)
    L_VAE = L_recon + β · D_KL(q_φ(z|X) || p(z))
    """

    def __init__(self, input_dim: int = NUM_PITCHES, hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM, num_layers: int = VAE_NUM_LAYERS,
                 seq_len: int = SEQUENCE_LENGTH, dropout: float = VAE_DROPOUT,
                 num_genres: int = NUM_GENRES, beta: float = VAE_BETA):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, num_layers, seq_len, dropout, num_genres)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ ⊙ ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor, genre: torch.Tensor | None = None):
        """
        Returns:
            x_hat, mu, logvar, z
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, genre)
        return x_hat, mu, logvar, z

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """D_KL(q(z|X) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)"""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    def loss_function(self, x: torch.Tensor, x_hat: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor,
                      beta: float | None = None) -> dict[str, torch.Tensor]:
        """Compute VAE loss: L_VAE = L_recon + β * D_KL"""
        if beta is None:
            beta = self.beta

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = self.kl_divergence(mu, logvar)
        total_loss = recon_loss + beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def generate(self, num_samples: int = 1, genre: torch.Tensor | None = None,
                 device: torch.device = DEVICE) -> torch.Tensor:
        """Generate new music by sampling z ~ N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        if genre is not None:
            genre = genre.to(device)
        with torch.no_grad():
            return self.decoder(z, genre)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    num_steps: int = 10) -> list[torch.Tensor]:
        """Latent space interpolation between two music sequences."""
        with torch.no_grad():
            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)
            results = []
            for alpha in torch.linspace(0, 1, num_steps):
                z = (1 - alpha) * mu1 + alpha * mu2
                x_hat = self.decoder(z)
                results.append(x_hat)
            return results
