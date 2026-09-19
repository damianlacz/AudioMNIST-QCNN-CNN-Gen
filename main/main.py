if __name__ == "__main__":
    import torch as t
    from torch.utils.data import DataLoader
    from main import dataset
    from models.vae import VAE
    from models.ddpm import DDPM
    from models.gan import GAN

