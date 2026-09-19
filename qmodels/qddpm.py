import torch as t
import torch.nn as nn
import numpy as np
import pennylane as qml

from models.ddpm import DiffusionModel
#from circuits import ddpm_circuit
from circuits import visualize_circuits

class QDDPM(DiffusionModel):
    def __init__(self, start=1e-4, end=0.02, n_steps=1000, time_dim=64, embed_dim=64):
        super(QDDPM, self).__init__(start=start, end=end, n_steps=n_steps, time_dim=time_dim, embed_dim=embed_dim)
        #self.ddpm_circuit = ddpm_circuit

    def forward(self, x):
        pass

    @t.no_grad()
    def generate_sample(self, batch_size=4, device='cpu') -> t.Tensor:
        return super(QDDPM, self).generate_sample(batch_size=batch_size, device=device)

    @t.no_grad()
    def visualize_sample(self, test_dataloader, batch_size=4, device='cpu') -> tuple[np.ndarray, np.ndarray]:
        orig_spec, gen_spec = super(QDDPM, self).visualize_sample(test_dataloader, batch_size=batch_size, device=device)
        visualize_circuits(self.ddpm_circuit)
        return orig_spec, gen_spec