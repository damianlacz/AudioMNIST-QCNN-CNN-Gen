import torch as t
import torch.nn as nn
import numpy as np
import pennylane as qml

from models.ddpm import DiffusionModel
#from circuits import ddpm_circuit
from circuits import visualize_circuits

class QuantumStepBottleneck(nn.Module):
    def __init__(self, time_dim=128, n_qubits=4):
        super(QuantumStepBottleneck, self).__init__()
        self.time_dim = time_dim
        self.n_qubits = n_qubits

        # Encoders and projections
        self.in_latent = nn.Linear(128 * 10 * 11, self.n_qubits)
        self.quantum_time_proj = nn.Linear(self.time_dim, self.n_qubits)
        self.out_latent = nn.Linear(self.n_qubits, 128 * 10 * 11)

        dev = qml.device("default.qubit", wires=self.n_qubits)

        weights = {"weights": (self.n_qubits,)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_forward_step(inputs, weights):
            x = inputs[:, :self.n_qubits]
            time_emb = inputs[:, self.n_qubits:]

            for i in range(self.n_qubits):
                qml.RY(x[:, i], wires=i)

            for i in range(self.n_qubits):
                qml.RX(time_emb[:, i], wires=i)
                qml.RY(weights[i], wires=i)

            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_inverse_step(inputs, weights):
            x = inputs[:, :self.n_qubits]
            time_emb = inputs[:, self.n_qubits:]

            for i in range(self.n_qubits):
                qml.RY(x[:, i], wires=i)

            qml.CNOT(wires=[self.n_qubits - 1, 0])
            for i in reversed(range(self.n_qubits - 1)):
                qml.CNOT(wires=[i, i + 1])

            for i in reversed(range(self.n_qubits)):
                qml.RY(-weights[i], wires=i)
                qml.RX(-time_emb[:, i], wires=i)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        self.circuits = dict(qml_forward=qml.qnn.TorchLayer(quantum_forward_step, weights), qml_inverse=qml.qnn.TorchLayer(quantum_inverse_step, weights))

    def forward(self, x, t_emb):
        angles = self.in_latent(x) * t.pi
        t_emb_angles = self.quantum_time_proj(t_emb) * t.pi
        inputs = t.cat([angles, t_emb_angles], dim=1)
        x = self.circuits["qml_inverse"](inputs)
        x = self.out_latent(x)
        return x.view(-1, 128, 10, 11)

class QuantumDiffusionModel(DiffusionModel):
    def __init__(self, start=1e-4, end=0.02, n_steps=1000, time_dim=64, embed_dim=64, n_qubits=4):
        super(QuantumDiffusionModel, self).__init__(start=start, end=end, n_steps=n_steps, time_dim=time_dim, embed_dim=embed_dim)
        self.n_qubits = n_qubits
        
        self.quantum_step_bottleneck = QuantumStepBottleneck(time_dim=time_dim+embed_dim, n_qubits=n_qubits)
        self.circuits = self.quantum_step_bottleneck.circuits

    def forward(self, x, time, label=None):

        x, t_emb, skips = self.model.encode(x, time, label=label)

        x = self.model.res_block(x, t_emb)

        x = self.quantum_step_bottleneck(x.flatten(1), t_emb)

        x = self.model.decode(x, t_emb, *skips)

        return self.model.out(x)

    def visualize_sample(self, test_dataset, batch_size=4, style="mpl", device='cpu', **kwargs):
        visualize_circuits(self.circuits, t.randn(batch_size, 2 * self.n_qubits), style=style, **kwargs)
        return super(QuantumDiffusionModel, self).visualize_sample(test_dataset, batch_size, device)