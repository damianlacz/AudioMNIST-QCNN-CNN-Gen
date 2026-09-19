import torch as t
import torch.nn as nn
import numpy as np
import pennylane as qml

from models.vae import VAE
#from circuits import vae_circuit
from circuits import visualize_circuits

class QVAE(VAE):
    def __init__(self, input_dim=128, latent_dim=16, hidden_dim=64):
        super(QVAE, self).__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        #self.vae_circuit = vae_circuit

    def reparameterize(self, mu, logvar):
        return super(QVAE, self).reparameterize(mu, logvar)

    def forward(self, x):
        pass

    @t.no_grad()
    def generate_sample(self, batch_size=4, device='cpu') -> t.Tensor:
        return super(QVAE, self).generate_sample(batch_size=batch_size, device=device)

    @t.no_grad()
    def visualize_sample(self, test_dataloader, batch_size=4, device='cpu') -> tuple[np.ndarray, np.ndarray]:
        orig_spec, gen_spec = super(QVAE, self).visualize_sample(test_dataloader, batch_size=batch_size, device=device)
        visualize_circuits(self.vae_circuit)
        return orig_spec, gen_spec
