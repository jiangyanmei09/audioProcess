# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：specAugment.py
@Author  ：jiangym
@Date    ：2023/4/12 上午9:07
@Description :torchaudio is the main tool to implement audio spectrogram augmentation including
            timeStretch, timeMasking and frequencyMasking
'''
import librosa
import torchaudio.sox_effects
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt

def _get_sample(path, resample=None):
    """
    :param path: 文件路径
    :param resample:重采样率
    :return:返回重采样后的声音文件
    """
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_speech_sample(path, resample=None):
    waveform, _ = _get_sample(path, resample=resample)
    return waveform


def get_spectrogram(waveform, n_fft=400, win_len=None, hop_length=None, power=2.0):
    """
    :param waveform: 声音信号
    :param n_fft: size of fft, create n_fft // 2 bins
    :param win_len: window size(窗长)，默认为n_fft，该值会影响FFT的时间分辨率和频率分辨率
    :param hop_length:帧移
    :param power:power=2为功率谱，power=1为能量谱
    :return:返回Tensor的spectrogram,若power=None时，返回的是torch.complex64，若power=2.0时，返回的是torch.float32
    """
    spectrogram = T.Spectrogram(
        n_fft=n_fft, win_length=win_len, hop_length=hop_length,
        center=True, pad_mode="reflect", power=power,
    )
    return spectrogram(waveform)

# TODO:增加f_min和f_max参数
def get_melspectrogram(waveform, sample_rate=16000, n_fft=400,
                       win_len=None, hop_length=None, n_mels=128,
                       power=2.0, normalized=True):
    """
    :param waveform: 声音信号
    :param sample_rate: 采样率
    :param n_fft: fft的点数
    :param win_len: 窗长，影响fft的频率和时域分辨率
    :param hop_length: 帧移
    :param n_mels:mel bins
    :param power:power=2为功率谱，power=1为能量谱
    :param normalized:Whether to normalize by magnitude after stft
    :return:
    """
    melspectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                      win_length=win_len, hop_length=hop_length,
                                      n_mels=n_mels, power=power, normalized=normalized)
    return melspectrogram(waveform)


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    """
    :param spec: spectrogram
    :param title: 图像的标题
    :param ylabel: y轴坐标
    :param aspect:
    :param xmax: x轴的最大值
    :return:
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def specTimeStretch(spec, n_filter=201, rate=None):
    """
    :param spec:tensor类型，torchaudio.transforms.Spectrogram()
    :param n_filter: number of filter banks from stft,for mel-spectrogram,it is mel bins.
    :param rate:rate to speed up or slow down by.加速 or 放慢
    :return: 加速 or 放慢的声音信号
    """
    stretch = T.TimeStretch(n_freq=n_filter)
    spec_ = stretch(spec, rate)
    return spec_[0]


def specTimeMasking(spec, seed=0, time_mask_param=None):
    """
    :param spec: tensor类型的，torchaudio.transforms.Spectrogram()
    :param seed: random seed
    :param time_mask_param: maximum possible length of the mask
                            Indices uniformly sampled from [0, time_mask_param)
    :return:
    """
    torch.random.manual_seed(seed)
    masking = T.TimeMasking(time_mask_param)
    spec = masking(spec)
    return spec[0]


def specFrequencyMasking(spec, seed=0, freq_mask_param=None):
    """
    :param spec:tensor类型，torchaudio.transforms.Spectrogram()
    :param seed: random seed
    :param freq_mask_param: maximum possible length of the mask.
                            Indices uniformly sampled from [0, freq_mask_param).
    :return:
    """
    torch.random.manual_seed(seed)
    masking = T.FrequencyMasking(freq_mask_param)
    spec = masking(spec)
    return spec[0]


if __name__ == "__main__":
    filepath = "/home/jym/PycharmProjects/audioProcess/data/1-137-A-32.wav"
    waveform, sample_rate = torchaudio.load(filepath)


# -------------------timeStretch-------------------------
    rate=1.5
    spec = get_spectrogram(waveform, n_fft=2048, win_len=1024, hop_length=256, power=None)
    spec_time_stretch = specTimeStretch(spec, n_filter=1025, rate=rate)

    # mel-spectrogram n_mels和n_filter的值必须保持一致
    melspec = get_melspectrogram(waveform, sample_rate=sample_rate, n_fft=400, win_len=None,
                                 hop_length=512,  n_mels=64)
    melspec_time_stretch = specTimeStretch(melspec, n_filter=64, rate=rate)


# ----------------timeMasking-------------------------
    spec1 = get_spectrogram(waveform)
    spec_time_masking = specTimeMasking(spec1, seed=50, time_mask_param=80)

    # mel-spectrogram
    melspec1 = get_spectrogram(waveform)
    melspec_time_masking = specTimeMasking(melspec1, seed=50, time_mask_param=80)


# ----------------frequencyMasking-------------------------
    spec_frequency_masking = specFrequencyMasking(spec1, seed=50, freq_mask_param=80)
    melspec_frequency_masking = specFrequencyMasking(melspec1, seed=50, freq_mask_param=80)



    # plot_spectrogram(torch.abs(spec_time_stretch), title=f"Spectrogram: Stretched x{rate}", aspect="equal", xmax=304)
    # plot_spectrogram(torch.abs(spec_time_masking),title="Spectrogram: Masked along time axis")
    # plot_spectrogram(torch.abs(spec_frequency_masking), title="Spectrogram: Masked along frequency axis")

    plot_spectrogram(torch.abs(melspec_time_stretch), title=f"MelSpectrogram: Stretched x{rate}", aspect="equal", xmax=304)
    plot_spectrogram(torch.abs(melspec_time_masking), title="MelSpectrogram: Masked along time axis")
    plot_spectrogram(torch.abs(melspec_frequency_masking), title="MelSpectrogram: Masked along frequency axis")







