import torch as t
import torch.nn as nn

class HiFiGAN(nn.Module):
    DEFAULT_CHECKPOINT = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/files/hifigan_gen_checkpoint_10000_ft.pt"

    def __init__(self, device='cpu'):
        super(HiFiGAN, self).__init__()
        self.generator, self.train_setup, self.denoiser = t.hub.load(repo_or_dir='nvidia/DeepLearningExamples:torchhub', model='nvidia_hifigan', source='github', verbose=True)
        self.device = device

    def _load_from_checkpoint(self, checkpoint=None):
        """Load the HiFi-GAN generator checkpoint."""
        self.checkpoint = checkpoint if checkpoint is not None else self.DEFAULT_CHECKPOINT
        self.state_dict_ = t.hub.load_state_dict_from_url(url=self.checkpoint, map_location=self.device)
        self.generator.load_state_dict(self.state_dict_['generator']).to(self.device)

    def forward(self, spec):
        return self.generator(spec)

    def fine_tune(self, lr=1e-6):
        pass

    def save_checkpoint(self, path=None):
        """Save the HiFi-GAN generator checkpoint."""
        if path is not None:
            t.save(self.generator.state_dict(), path)