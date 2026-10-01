import torch as t
import torch.nn as nn
import numpy as np
import pennylane as qml

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.vae import VAE
from circuits import visualize_circuits


class QVAE(VAE):
    """
    Quantum Variational Autoencoder inheriting directly from VAE.
    Uses PennyLane Quantum Circuits to form a Quantum Latent Space (Density Matrix).
    """
    default_recon_loss = nn.MSELoss()

    @staticmethod
    def default_kl_loss(rho, prior_rho=None, eps=1e-8):
        """
        Calculates Quantum Relative Entropy (Quantum KL Divergence) on density matrices:
        D_KL(rho || sigma) = Tr(rho * log(rho) - rho * log(sigma))
        """
        eigs_rho = t.clamp(t.linalg.eigvalsh(rho), min=eps)
        tr_rho_log_rho = t.sum(eigs_rho * t.log(eigs_rho), dim=-1)

        if prior_rho is None:
            matrix_dim = rho.shape[-1]
            kl = tr_rho_log_rho + t.log(matrix_dim)
        else:
            eigs_prior = t.clamp(t.linalg.eigvalsh(prior_rho), min=eps)
            tr_rho_log_prior = t.sum(eigs_rho * t.log(eigs_prior), dim=-1)
            kl = tr_rho_log_rho - tr_rho_log_prior

        return t.mean(kl)

    def __init__(self, num_classes=10, embed_dim=64, n_qubits=4, n_layers=2):
        super(QVAE, self).__init__(latent_dim=n_qubits, num_classes=num_classes, embed_dim=embed_dim)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        #self.quantization_layer = nn.Linear(latent_dim, n_qubits)
        self.encoder.bneck2.out_features = n_qubits


        #self.unquantization_layer = nn.Linear(2 * self.dim * self.dim, latent_dim)
        self.decoder.bneck = nn.Linear(2 * self.dim * self.dim, self.decoder.bneck.out_features)

        weigths = nn.Parameter(t.randn(n_layers, n_qubits, 3) * 0.1)
        dev = qml.device("default.mixed", wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def default_quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")

            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            for i in range(self.n_qubits):
              qml.DepolarizingChannel(p=0.05, wires=i)

            return qml.density_matrix(wires=range(self.n_qubits))

        self.circuits = dict(qkl_circuit=qml.qnn.TorchLayer(default_quantum_circuit, weigths))

    def quantum_encode(self, x):
        feat, _, _ = self.encoder(x)
        angles = t.pi * t.tanh(feat)
        rho = self.circuits["qkl_circuit"](angles, self.q_weights)
        return rho

    def forward(self, x, label=None):
        rho = self.quantum_encode(x)

        z = t.cat([rho.real.flatten(1), rho.imag.flatten(1)], dim=1).float()

        #z = self.unquantization_layer(rho_cat)
        emb = self.num_embed(label) if label is not None else None
        reconstruction = self.decoder(z, emb)

        return reconstruction, rho

    def fit(self, dataloader, optimizer, recon_loss=None, kl_loss=None, include_labels=True, alpha=1.0, beta=1.0, epochs=10, device=t.device('cpu')):
        recon_loss = recon_loss or self.default_recon_loss
        kl_loss = kl_loss or self.default_kl_loss
        super(QVAE, self).fit(dataloader, optimizer, recon_loss=recon_loss, kl_loss=kl_loss, include_labels=include_labels, alpha=alpha, beta=beta, epochs=epochs, device=device)

    @t.no_grad()
    def sample(self, label=None, batch_size=4, device='cpu'):
        indentity = t.eye(self.dim, device=device)
        sigma_real = (indentity.unsqueeze(0).repeat(batch_size, 1, 1) / self.dim).flatten(1)
        sigma_imag = t.zeros_like(sigma_real)
        sigma = t.cat([sigma_real, sigma_imag], dim=1).float()

        #z = self.unquantization_layer(sigma)

        if label is None:
            label = t.randint(0, self.num_classes, (batch_size,), device=device)

        emb = self.num_embed(label)
        return sigma, emb

    @t.no_grad()
    def generate_sample(self, label=None, batch_size=4, device='cpu'):
        self.eval()
        z, emb = self.sample(label=label, batch_size=batch_size, device=device)
        return self.decoder(z, emb)

    @t.no_grad()
    def visualize_sample(self, test_dataset, batch_size=4, device='cpu', style="mpl", **kwargs):
        self.eval()
        sample, label = next(iter(DataLoader(test_dataset, batch_size=batch_size, shuffle=True)))

        gen = self.generate_sample(label=label, batch_size=batch_size, device=device).cpu().numpy()
        recon = self(sample, label)[0].cpu().numpy()
        sample = sample.cpu().numpy()

        visualize_circuits(self.circuits, t.zeros(self.n_qubits, device=device), style=style, kwargs=kwargs)
        return super(QVAE, self).visualize_spectrograms(sample, gen, recon, label, batch_size=batch_size)
