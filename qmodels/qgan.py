import torch as t
import torch.nn as nn
import numpy as np
import pennylane as qml

import models.gan as gan
import qmodels.circuits as circuits

from models.gan import GAN, Generator, Discriminator
from qmodels.circuits import generator_circuit, discriminator_circuit
from qmodels.circuits import visualize_circuits

from tqdm import tqdm

import torch as t
import torch.nn as nn
import pennylane as qml
import numpy as np

class QGenerator(Generator):
    def __init__(self, n_qubits=2, n_qubits_label=2, n_layers=3):
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.n_qubits_label = n_qubits_label

        self.dm_dim = (2 ** self.n_qubits) ** 2

        super(QGenerator, self).__init__(noise_dim=self.dm_dim, embed_dim=n_qubits_label)

        self.total_qubits = self.n_qubits + self.n_qubits_label

        dev = qml.device("default.qubit", wires=self.total_qubits)
        weights = {"weights": (self.n_layers, self.total_qubits, 3)}
        @qml.qnode(dev, interface='torch', diff_method="backprop")
        def default_quantum_generator_circuit(inputs, weights):
            angles = inputs[:, :self.n_qubits]
            emb_angles = inputs[:, self.n_qubits:]

            for i in range(self.n_qubits):
                qml.RY(angles[:, i], wires=i)

            for i in range(self.n_qubits_label):
                qml.RY(emb_angles[:, i], wires=self.n_qubits + i)

            for layer in range(self.n_layers):
                for i in range(self.total_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                for i in range(self.total_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.total_qubits])

            return qml.density_matrix(wires=range(self.n_qubits))

        self.circuits = dict(qml_generator=qml.qnn.TorchLayer(default_quantum_generator_circuit, weights))

    def forward(self, x, label=None, verbose=False):
        z = t.tanh(x) * t.pi
        emb = t.tanh(self.num_embed(label)) * t.pi

        inputs = t.cat([z, emb], dim=1)
        rho = self.circuits["qml_generator"](inputs)
        return super(QGenerator, self).forward(rho.flatten(1), label=label, verbose=verbose)

class QDiscriminator(Discriminator):
    def __init__(self, n_qubits=2, n_qubits_label=2, n_layers=3):
        super(QDiscriminator, self).__init__(embed_dim=n_qubits_label)
        self.n_qubits = n_qubits
        self.n_qubits_label = n_qubits_label
        self.n_layers = n_layers
        self.dm_dim = (2 ** self.n_qubits, 2 ** self.n_qubits)
        self.total_qubits = n_qubits + n_qubits_label + 1

        self.dense = nn.Linear(self.dense.in_features, 2 ** self.n_qubits)
        self.dense_uncond = nn.Linear(self.dense_uncond.in_features, 2 ** self.n_qubits)

        dev = qml.device("default.mixed", wires=self.total_qubits)
        weights = {"weights": (n_layers, self.total_qubits, 3)}
        @qml.qnode(dev, interface='torch', diff_method="backprop")
        def default_discriminator_circuit(inputs, weights):
            sep = int(np.prod(self.dm_dim))
            rho = inputs[:sep].reshape(*self.dm_dim)
            emb_angles = inputs[sep:]

            # TODO: MAKE IT BATCHED
            dm = qml.QubitDensityMatrix(rho, wires=range(self.n_qubits))

            for i in range(self.n_qubits_label):
                qml.RY(emb_angles[i], wires=self.n_qubits+i)

            for layer in range(self.n_layers):
                for i in range(self.total_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

                for i in range(self.total_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.total_qubits])

            return qml.expval(qml.PauliZ(wires=self.total_qubits-1))

        self.circuits = dict(qml_discriminator=qml.qnn.TorchLayer(default_discriminator_circuit, weights))

    def forward(self, rho, label=None):
        emb = t.tanh(self.num_embed(label)) * t.pi
        inputs = t.cat([rho.flatten(1), emb], dim=1)
        expvals = t.stack([self.circuits["qml_discriminator"](inputs[b]) for b in range(rho.shape[0])])
        return (1.0 + expvals) / 2.0

class QGAN(GAN):
    def __init__(self, n_qubits=2, n_qubits_label=2, n_layers_gen=3, n_layers_disc=3):
        nn.Module.__init__(self)
        self.noise_dim = n_qubits
        self.generator = QGenerator(n_qubits=n_qubits, n_qubits_label=n_qubits_label, n_layers=n_layers_gen)
        self.discriminator = QDiscriminator(n_qubits=n_qubits, n_qubits_label=n_qubits_label, n_layers=n_layers_disc)

        weights = {"weights": (n_layers_disc, self.noise_dim, 3)}
        dev = qml.device("default.qubit", wires=self.noise_dim)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_encoder(inputs, weights):
          qml.AmplitudeEmbedding(inputs, wires=range(self.noise_dim), normalize=True)
          for layer in range(n_layers_disc):
              for i in range(self.noise_dim):
                  qml.Rot(
                      weights[layer, i, 0],
                      weights[layer, i, 1],
                      weights[layer, i, 2],
                      wires=i
                  )

              for i in range(self.noise_dim - 1):
                  qml.CNOT(wires=[i, i + 1])

          return qml.density_matrix(wires=range(self.noise_dim))

        self.quantum_encoder = qml.qnn.TorchLayer(quantum_encoder, weights)

        self.circuits = self.generator.circuits | self.discriminator.circuits

    def forward(self, spec, label=None):
        noise = t.randn(spec.shape[0], self.noise_dim, device=spec.device)

        gen_spec, rho = self.generator(noise, label=label, verbose=True)
        gen_prob = self.discriminator(rho, label=label)

        amplitude = self.discriminator.spec_encoder(spec, label=label)
        #print(amplitude)
        sigma = self.quantum_encoder(amplitude)
        spec_prob = self.discriminator(sigma, label=label)

        return gen_spec, gen_prob, spec_prob
    
    def calc_gen_loss(self, orig_spec, gen_spec, gen_loss_fn=None, disc_loss_fn=None, label=None, alpha=1.0, beta=1.0, device='cpu', **kwargs):
      return super(QGAN, self).calc_gen_loss(orig_spec, gen_spec, gen_loss_fn=gen_loss_fn, disc_loss_fn=disc_loss_fn, label=label, alpha=alpha, beta=beta, device=device, **kwargs)

    def calc_disc_loss(self, orig_spec, gen_spec, disc_loss_fn=None, label=None, beta=1.0, label_smoothing=0.8, device='cpu', **kwargs):
      amplitude = self.discriminator.spec_encoder(orig_spec, label=label)
      sigma, rho = self.quantum_encoder(amplitude), gen_spec
      return super(QGAN, self).calc_disc_loss(sigma, rho, disc_loss_fn=disc_loss_fn, label=label, beta=beta, label_smoothing=label_smoothing, device=device, **kwargs)

    def visualize_sample(self, test_dataset, label=None, batch_size=4, device='cpu', style='mpl', **kwargs):
        dim = np.prod(self.generator.dm_dim) + self.generator.n_qubits_label
        visualize_circuits(self.generator.circuits, t.randn(size=(batch_size, dim)), style=style, **kwargs)
        visualize_circuits(self.discriminator.circuits, t.randn(size=(batch_size, dim)), style=style, **kwargs)
        return super(QGAN, self).visualize_sample(test_dataset, label=label, batch_size=batch_size, device=device)
