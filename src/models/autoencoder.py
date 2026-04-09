"""
Task 1: LSTM Autoencoder for Single-Genre Music Generation.

Architecture:
    Encoder: Bidirectional LSTM that maps input sequence X to latent vector z = f_phi(X)
    Decoder: LSTM that reconstructs X_hat = g_theta(z) from z

Loss: L_AE = sum_t || x_t - x_hat_t ||^2
"""
import torch
import torch.nn as nn

from src.config import (
    NUM_PITCHES, AE_LATENT_DIM, AE_HIDDEN_DIM, AE_NUM_LAYERS,
    AE_DROPOUT, SEQUENCE_LENGTH, DEVICE,
)


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder: maps piano-roll sequence to latent vector z."""

    def __init__(self, input_dim: int = NUM_PITCHES, hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM, num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) piano-roll input
        Returns:
            z: (batch, latent_dim) latent embedding
        """
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers*2, batch, hidden_dim)
        # Concatenate forward and backward final hidden states from last layer
        h_forward = h_n[-2]   # (batch, hidden_dim)
        h_backward = h_n[-1]  # (batch, hidden_dim)
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden_dim*2)
        z = self.fc_latent(h_cat)  # (batch, latent_dim)
        return z


class LSTMDecoder(nn.Module):
    """LSTM decoder: reconstructs piano-roll sequence from latent vector z."""

    def __init__(self, latent_dim: int = AE_LATENT_DIM, hidden_dim: int = AE_HIDDEN_DIM,
                 output_dim: int = NUM_PITCHES, num_layers: int = AE_NUM_LAYERS,
                 seq_len: int = SEQUENCE_LENGTH, dropout: float = AE_DROPOUT):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_init = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_hat: (batch, seq_len, output_dim) reconstructed sequence
        """
        batch_size = z.size(0)

        # Initialize hidden / cell states from z
        h_0 = self.fc_init(z).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = self.fc_cell(z).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Repeat z across time steps as decoder input
        decoder_input = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)

        output, _ = self.lstm(decoder_input, (h_0, c_0))  # (batch, seq_len, hidden_dim)
        x_hat = torch.sigmoid(self.fc_out(output))  # (batch, seq_len, output_dim)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """Complete LSTM Autoencoder for music generation (Task 1).

    z = f_phi(X)        (encoder)
    X_hat = g_theta(z)  (decoder)
    L_AE = ||X - X_hat||^2
    """

    def __init__(self, input_dim: int = NUM_PITCHES, hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM, num_layers: int = AE_NUM_LAYERS,
                 seq_len: int = SEQUENCE_LENGTH, dropout: float = AE_DROPOUT):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers, seq_len, dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            x_hat: (batch, seq_len, input_dim)
            z: (batch, latent_dim)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """L_AE = sum_t ||x_t - x_hat_t||^2  (MSE)"""
        return nn.functional.mse_loss(x_hat, x, reduction="mean")

    def generate(self, num_samples: int = 1, device: torch.device = DEVICE) -> torch.Tensor:
        """Generate new music by sampling random latent codes."""
        z = torch.randn(num_samples, self.encoder.fc_latent.out_features).to(device)
        with torch.no_grad():
            return self.decode(z)
