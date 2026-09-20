import os
import librosa
import numpy as np
import torch as t
import torchaudio
import urllib
import zipfile

from torch.utils.data import Dataset

class AudioMNISTDataset(Dataset):
    def __init__(self, base_dir="./data/AudioMNIST", max_folders=20, sr=22050):
      super().__init__()

      self.base_dir = base_dir
      self.sr = sr

      self.n_mels = 80
      self.n_fft = 1024
      self.win_length = 1024
      self.hop_length = 256
      self.clamp_value = 1e-5
      self.norm_value = 10.0

      self.file_list = []
      self.labels = []

      if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Downloading dataset from github...")
        self._download_dataset()

      all_folders = sorted([
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()
      ])

      selected_folders = all_folders[:max_folders]

      print(f"Loading data from {len(selected_folders)} speakers...")

      for folder in selected_folders:
        folder_path = os.path.join(base_dir, folder)
        for filename in os.listdir(folder_path):
          if filename.endswith('.wav'):
            digit = int(filename.split('_')[0])
            self.file_list.append(os.path.join(folder_path, filename))
            self.labels.append(digit)

      self.mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=self.sr,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.win_length,
        n_mels=self.n_mels,
        center=True,
        pad_mode="reflect",
        power=1.0,
        normalized=True
      )

      self.inv_mel_transform = torchaudio.transforms.InverseMelScale(
        n_mels=self.n_mels,
        sample_rate=self.sr,
        n_stft=self.n_fft // 2 + 1
      )

      self.griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.win_length,
        n_iter=64
      )

    def _download_dataset(self):
      os.makedirs(self.base_dir, exist_ok=True)
      dirname = os.path.dirname(self.base_dir)
      path = os.path.join(dirname, "AudioMNIST.zip")
      urllib.request.urlretrieve("https://github.com/soerenab/AudioMNIST/archive/refs/heads/master.zip", path)
      with zipfile.ZipFile(path, "r") as zfile:
        zfile.extractall(dirname)

      extracted_dir = os.path.dirname(dirname, "AudioMNIST-master")
      for item in os.listdir(extracted_dir):
        src, dst = os.path.join(extracted_dir, item), os.path.join(self.base_dir, item)
        if os.path.isdir(src) and item.isdigit():
          os.rename(src, dst)

      os.remove(path)

      try: 
        os.rmdir(extracted_dir) 
      except OSError: 
        pass

      print(f"Downloaded AudioMNIST dataset to directory {self.base_dir}")
      return

    def __len__(self):
      return len(self.file_list)

    def __getitem__(self, idx):
      file_path = self.file_list[idx]
      label = self.labels[idx]

      audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
      audio = np.pad(audio, (0, self.sr - len(audio)), mode='constant') if len(audio) < self.sr else audio[:self.sr]
      audio = t.tensor(audio, dtype=t.float32)

      mel_spec = self.mel_transform(audio)
      mel_spec = (t.log(t.clamp(mel_spec, min=self.clamp_value)) - t.log(t.tensor(self.clamp_value, dtype=t.float32))) / self.norm_value
      #mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
      #mel_spec = mel_spec - self.log_min / (self.log_max - self.log_min)

      return mel_spec.unsqueeze(0), t.tensor(label, dtype=t.long)

    @t.no_grad()
    def spec_to_audio_griffin(self, spec):
      """Convert spectrogram to audio using Griffin-Lim algorithm."""
      spec = spec * self.norm_value + t.log(t.tensor(self.clamp_value, dtype=t.float32))
      inv_spec = self.inv_mel_transform(spec)
      audio = self.griffin_lim(inv_spec)
      return librosa.util.normalize(audio.cpu().numpy())

    @t.no_grad()
    def spec_to_audio_hifi(self, spec, hifigan):
      """Convert spectrogram to audio using HiFi-GAN vocoder."""
      hifigan.eval()
      #spec = spec * (spec.max() - spec.min()) + spec.min()
      #spec = t.log(t.clamp(spec, min=self.clamp_value))
      spec = t.tensor(spec, dtype=t.float32)
      spec = spec * self.norm_value + t.log(t.tensor(self.clamp_value, dtype=t.float32))
      audio = hifigan(spec)
      audio = audio.squeeze().cpu().numpy()
      return librosa.util.normalize(audio)