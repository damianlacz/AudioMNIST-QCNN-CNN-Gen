import torch as t
import torch.nn as nn
import numpy as np
import pennylane as qml

import models.gan as gan
import qmodels.circuits as circuits

from models.gan import GAN
from qmodels.circuits import generator_circuit, discriminator_circuit
from qmodels.circuits import visualize_circuits

from tqdm import tqdm

class QGAN(GAN):
    def __init__(self, noise_dim=16, gen_embed_dim=64, disc_embed_dim=64, gen_lr=1e-4, disc_lr=1e-3):
        super(QGAN, self).__init__(noise_dim=noise_dim, gen_embed_dim=gen_embed_dim, disc_embed_dim=disc_embed_dim, gen_lr=gen_lr, disc_lr=disc_lr)
        self.generator_circuit = generator_circuit
        self.discriminator_circuit = discriminator_circuit

    def forward(self, x, label):
        super(QGAN, self).forward(x, label)


    @t.no_grad()
    def generate_sample(self, label=None, batch_size=4, device='cpu') -> t.Tensor:
        return super(QGAN, self).generate_sample(label=label, batch_size=batch_size, device=device)

    @t.no_grad()
    def visualize_sample(self, test_dataloader, batch_size=4, device='cpu') -> tuple[np.ndarray, np.ndarray]:
        orig_spec, gen_spec = super(QGAN, self).visualize_sample(test_dataloader, batch_size=batch_size, device=device)
        visualize_circuits(self.generator_circuit, self.discriminator_circuit)
        return orig_spec, gen_spec
