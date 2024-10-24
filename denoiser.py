import torch
import numpy as np


class Denoiser(torch.nn.Module):
    def __init__(self,
                 thr: float = 1.0,
                 red_rate: float = 1.1,
                 n_fft: int = 1024,
                 hop_length=None,
                 win_length=None) -> None:
        super(Denoiser, self).__init__()
        self.thr = thr
        self.red_rate = red_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length else n_fft // 4
        self.win_length = win_length if win_length else n_fft
        self.noise_profile = None

    def fit(self, noise_sample) -> None:
        """
        Estimate the noise profile from a sample of noise.
        :param noise_sample: 1D numpy array or torch tensor containing the noise audio.
        """
        if isinstance(noise_sample, np.ndarray):
            noise_sample = torch.from_numpy(noise_sample)
        noise_sample = noise_sample.float()

        noise_stft = torch.stft(
            noise_sample,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )

        noise_mag = torch.abs(noise_stft)

        self.noise_profile = torch.mean(noise_mag, dim=1)
        self.noise_profile += 1e-6

    def forward(self, audio_wav):
        """
        Remove noise from the input audio using the estimated noise profile.
        :param audio_wav: 1D numpy array or torch tensor containing the audio.
        :return: Denoised audio as a torch tensor.
        """
        if self.noise_profile is None:
            raise ValueError("Noise profile has not been estimated. Please call fit() first.")

        if isinstance(audio_wav, np.ndarray):
            audio_wav = torch.from_numpy(audio_wav)
        audio_wav = audio_wav.float()

        audio_stft = torch.stft(
            audio_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )

        mag = torch.abs(audio_stft)
        phase = torch.angle(audio_stft)

        P = self.noise_profile.unsqueeze(1)
        S = mag

        S += 1e-6

        factor = 1 - (P * self.thr) / S
        factor = torch.clamp(factor, min=0.0, max=1.0)
        factor /= self.red_rate

        mag_denoised = mag * factor

        audio_stft_denoised = mag_denoised * torch.exp(1j * phase)

        audio_denoised = torch.istft(
            audio_stft_denoised,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=audio_wav.shape[-1]
        )

        return audio_denoised
