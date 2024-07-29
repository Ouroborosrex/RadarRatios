import numpy as np
from scipy.signal import chirp, gausspulse, butter, lfilter, hilbert
import torch
#from torchsig_main.torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer
from torchsig_main.torchsig.datasets.modulations import ModulationsDataset
#from torchsig_main.torchsig.utils.dataset import SignalDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
#import pickle
import scipy.signal as sp
import cv2
import random
#import lmdb

class torchsig_data:
    def __init__(self, num_samples=40000, num_iq_samples=1000, level = 0, include_snr = True, classes = None):
        self.num_samples = num_samples
        if classes is None:
            self.classes = [
                    "ook",
                    "bpsk",
                    "4pam",
                    "4ask",
                    "qpsk",
                    "8pam",
                    "8ask",
                    "8psk",
                    "16qam",
                    "16pam",
                    "16ask",
                    "16psk",
                    "32qam",
                    "32qam_cross",
                    "32pam",
                    "32ask",
                    "32psk",
                    "64qam",
                    "64pam",
                    "64ask",
                    "64psk",
                    "128qam_cross",
                    "256qam",
                    "512qam_cross",
                    "1024qam",
                    "2fsk",
                    "2gfsk",
                    "2msk",
                    "2gmsk",
                    "4fsk",
                    "4gfsk",
                    "4msk",
                    "4gmsk",
                    "8fsk",
                    "8gfsk",
                    "8msk",
                    "8gmsk",
                    "16fsk",
                    "16gfsk",
                    "16msk",
                    "16gmsk",
                    "ofdm-64",
                    "ofdm-72",
                    "ofdm-128",
                    "ofdm-180",
                    "ofdm-256",
                    "ofdm-300",
                    "ofdm-512",
                    "ofdm-600",
                    "ofdm-900",
                    "ofdm-1024",
                    "ofdm-1200",
                    "ofdm-2048",
                    ]
        else:
            self.classes = classes
        self.num_classes = len(self.classes)

        pl.seed_everything(1234567891)
        self.dataset = ModulationsDataset(
        classes=classes,
        use_class_idx=False,
        level=level,
        num_iq_samples=num_iq_samples+20,
        num_samples=int(self.num_classes * self.num_samples),
        include_snr=include_snr,)
        self.include_snr = include_snr
            # Seed the dataset instantiation for reproduceability
    def get_signal(self, signal_name):

        idx = random.randint(self.classes.index(signal_name) * self.num_samples, self.classes.index(signal_name) * self.num_samples + (self.num_samples-1))
        if self.include_snr:
            data, (modulation, snr) = self.dataset[idx]
        else:
            data, modulation = self.dataset[idx]
        data = data[:1000]
        return torch.tensor(data.astype(np.complex64)), modulation


def generate_barker_signal_iq_torch(length):
    barker_code = torch.tensor([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=torch.float)
    repetitions = length // barker_code.numel() + 1
    signal = barker_code.repeat(repetitions)[:length]
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_am_comm_signal_iq_torch(length):
    freq_carrier = 540e3
    modulating_freq = 1e3
    time = torch.linspace(0, length, length)
    modulating_signal = torch.sin(2 * np.pi * modulating_freq * time / time[-1])
    signal = torch.cos(2 * np.pi * freq_carrier * time / time[-1]) * (1 + modulating_signal)
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_rectangular_signal_iq_torch(length, duty_cycle=0.5):
    signal = torch.zeros(length)
    signal[:int(length * duty_cycle)] = 1
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_lfm_signal_iq_torch(length):
    start_freq = 76e9
    end_freq = 77e9
    duration = 5.5e-6
    time = torch.linspace(0, duration, length)
    signal = torch.tensor(chirp(time.numpy(), f0=start_freq, f1=end_freq, t1=duration, method='linear'), dtype=torch.float)
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_gfsk_signal_iq_torch(length, bitrate=1e6, deviation=250e3):
    baud_length = int(length / bitrate)
    time = torch.linspace(-baud_length / 2, baud_length / 2, length)
    signal = torch.tensor(gausspulse(time.numpy(), fc=0, bw=deviation), dtype=torch.float)
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_cpfsk_signal_iq_torch(length, bitrate=9600, deviation=5e3):
    time = torch.linspace(0, length/bitrate, length)
    data = torch.randint(0, 2, (length,)) * 2 - 1
    phase = torch.cumsum(data, dim=0) * 2 * np.pi * deviation / bitrate
    signal = torch.cos(phase)
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_bfm_signal_iq_torch(length, carrier_freq=2.4e9, mod_index=1.0, audio_freq=1e3):
    time = torch.linspace(0, length, length)
    audio_signal = torch.sin(2 * np.pi * audio_freq * time / time[-1])
    signal = torch.cos(2 * np.pi * carrier_freq * time / time[-1] + mod_index * audio_signal)
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_dsb_am_signal_iq_torch(length):
    carrier_freq = 1.8e6
    modulating_freq = 1e3
    time = torch.linspace(0, length, length)
    modulating_signal = torch.sin(2 * np.pi * modulating_freq * time / time[-1])
    signal = torch.cos(2 * np.pi * carrier_freq * time / time[-1]) * (1 + modulating_signal)
    iq_signal = signal + 1j * torch.tensor(np.imag(hilbert(signal.numpy())), dtype=torch.float)
    return iq_signal

def generate_16qam_signal_iq_torch(length):
    carrier_freq = 1.8e9
    symbol_rate = 15e3
    time = torch.linspace(0, length / symbol_rate, length)
    symbols = torch.tensor([-3-3j, -3-1j, -3+3j, -3+1j, -1-3j, -1-1j, -1+3j, -1+1j, 3-3j, 3-1j, 3+3j, 3+1j, 1-3j, 1-1j, 1+3j, 1+1j], dtype=torch.complex64)
    indices = torch.randint(0, len(symbols), (length,))
    qam_signal = symbols[indices]
    return torch.real(qam_signal) * torch.cos(2 * np.pi * carrier_freq * time) - torch.imag(qam_signal) * torch.sin(2 * np.pi * carrier_freq * time)

def mix_signals(signal1, signal2, mix_ratio):
    return mix_ratio * signal1 + (1 - mix_ratio) * signal2

def add_white_gaussian_noise(signal, snr):
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / 10 ** (snr / 10)
    noise = torch.randn_like(signal.real) + 1j * torch.randn_like(signal.imag)
    noise *= torch.sqrt(noise_power)
    return signal + noise

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if isinstance(data, torch.Tensor):
        if torch.is_complex(data):
            real_filtered = lfilter(b, a, data.real.numpy())
            imag_filtered = lfilter(b, a, data.imag.numpy())
            filtered_data = torch.tensor(real_filtered, dtype=torch.float) + 1j * torch.tensor(imag_filtered, dtype=torch.float)
        else:
            data = data.numpy()
            filtered_data = lfilter(b, a, data)
            filtered_data = torch.tensor(filtered_data)
    else:
        filtered_data = lfilter(b, a, data)
    return filtered_data

def noise_percentage_to_snr(noise_percentage):
    return -20 * np.log10(noise_percentage)

def generate_random_comm_signal(length, sig53_obj = None, comm_53_types=None):
    comm_signals = ['am_comm', 'cpfsk', 'bfm', 'dsb_am', 'qam']

    if comm_53_types is None:
        comm_53_types = []

    if sig53_obj is not None and comm_53_types != []:
        choices = comm_signals+comm_53_types
    else:
        choices = comm_signals

    choice = np.random.choice(choices)
    signal = None
    if choice == 'am_comm':
        signal = generate_am_comm_signal_iq_torch(length)
    elif choice == 'cpfsk':
        signal = generate_cpfsk_signal_iq_torch(length)
    elif choice == 'bfm':
        signal = generate_bfm_signal_iq_torch(length)
    elif choice == 'dsb_am':
        signal = generate_dsb_am_signal_iq_torch(length)
    elif choice == 'qam':
        signal = generate_16qam_signal_iq_torch(length)
    elif comm_53_types is not None and choice in comm_53_types:
        signal, modulation = sig53_obj.get_signal(choice)
    return signal, choice

def generate_random_radar_signal(length):
    radar_signals = ['barker', 'lfm', 'gfsk', 'rect']
    choice = np.random.choice(radar_signals)
    signal = None
    if choice == 'barker':
        signal = generate_barker_signal_iq_torch(length)
    elif choice == 'lfm':
        signal = generate_lfm_signal_iq_torch(length)
    elif choice == 'gfsk':
        signal = generate_gfsk_signal_iq_torch(length)
    elif choice == 'rect':
        signal = generate_rectangular_signal_iq_torch(length)
    return signal, choice

def generate_signals(length, snr, mix_ratio, num_samples = 1000, apply_bandpass=False):
    signals = np.zeros((num_samples, length), dtype=np.complex64)

    comm_types = ['am_comm', 'cpfsk', 'bfm', 'dsb_am', 'qam']
    comm_types_53 = ['4ask','8ask','8psk','16psk','16qam','64qam','2fsk','2gfsk','ofdm-64','ofdm-72']
    radar_types = ['barker', 'lfm', 'gfsk', 'rect']

    combined_comms_types = comm_types+comm_types_53

    sig53 = torchsig_data(classes=comm_types_53, num_samples=200)

    labels = np.zeros((num_samples, len(comm_types+comm_types_53+radar_types)))


    for i in range(num_samples):
        comm_signal, comm_type = generate_random_comm_signal(length, sig53, comm_types_53)
        radar_signal, radar_type = generate_random_radar_signal(length)
        comm_index = combined_comms_types.index(comm_type)
        radar_index = radar_types.index(radar_type) + len(combined_comms_types)

        mixed_signal = mix_signals(comm_signal, radar_signal, mix_ratio)
        noisy_signal = add_white_gaussian_noise(mixed_signal, snr)

        if apply_bandpass:
            filtered_signal = bandpass_filter(noisy_signal, 700, 800, 2000, order=5)
            labels[i, comm_index] = mix_ratio
            labels[i, radar_index] = 1 - mix_ratio
        else:
            filtered_signal = noisy_signal
            labels[i, comm_index] = mix_ratio
            labels[i, radar_index] = 1 - mix_ratio

        signals[i, :] = filtered_signal

    return signals, labels

import io
from PIL import Image

def spectrogram(data, fs=1.0, window=sp.windows.tukey(256, 0.25), nperseg=256, return_onesided=False):
    # Assuming iq_data is your combined IQ data
    # iq_data = np.array([...])
    _, _, spectrogram = sp.spectrogram(
                    x=data,
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    return_onesided=return_onesided,
                )
    spectrogram = 20 * np.log10(np.fft.fftshift(np.abs(spectrogram), axes=0))
    fig, ax = plt.subplots()
    # ax.figure()
    ax.imshow(
        spectrogram,
        vmin=np.min(spectrogram[spectrogram != -np.inf]),
        vmax=np.max(spectrogram[spectrogram != np.inf]),
        aspect="auto",
        cmap="jet",
    )
    ax.grid(False)  # Turn off grid
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks
    ax.axis('off')  # Hide axes border
    fig.tight_layout()
    fig.canvas.draw()
    ax.grid(False)  # Turn off grid

    img_plot = np.array(fig.canvas.buffer_rgba())
    # print(img_plot.shape)
    # plt.show()
    plt.close()
    return img_plot

from multiprocessing import Pool

def process_data(i, dataset, resize=(256,256)):
    if len(dataset) == 2:
        image = spectrogram(dataset[0][i])
        resized_img = cv2.resize(image, resize)
        return resized_img, dataset[1][i]
    else:
        image = spectrogram(dataset[i])
        resized_img = cv2.resize(image, resize)
        return resized_img, None


def twod_dataset(dataset, resize=(256,256)):
    with Pool() as p:
        if len(dataset) == 2:
            results = list(tqdm(p.starmap(process_data, [(i, dataset, resize) for i in range(len(dataset[0]))]), desc='Processing spectrograms',unit='spectrogram'))
            imgs, imgs_classes = zip(*results)
            return np.array(imgs)[:, :, :, :3], np.array(imgs_classes)
        else:
            results = list(tqdm(p.starmap(process_data, [(i, dataset, resize) for i in range(len(dataset))]), desc='Processing spectrograms',unit='spectrogram'))
            imgs, _ = zip(*results)
            return np.array(imgs, dtype=np.ubyte)[:, :, :, :3]


# def spec_dataset(dataset, resize_shape=(256, 256)):
#     imgs_classes = []
#
#     for i in range(len(dataset[0])):


datasets = {}
# combined_dataset_X = np.array()
# combined_dataset_y = np.array()

first = True

noise_levels = [0, 15, 30, 45, 60, 75]
mix_ratios = [0.0, 0.10, 0.20, 0.30, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]

for noise_percentage in noise_levels:
    snr = noise_percentage_to_snr(noise_percentage / 100.0)
    for mix_ratio in mix_ratios:
        signals_no_filter, labels_no_filter = generate_signals(1000, snr, mix_ratio, num_samples=3000, apply_bandpass=False)
        signals_with_filter, labels_with_filter = generate_signals(1000, snr, mix_ratio, num_samples=3000, apply_bandpass=True)
        if first:
            combined_dataset_X = np.array(signals_no_filter)
            combined_dataset_y = np.array(labels_no_filter)
            combined_dataset_X = np.append(combined_dataset_X, signals_with_filter, axis=0)
            combined_dataset_y = np.append(combined_dataset_y, labels_with_filter, axis=0)
            first = False
        else:
            combined_dataset_X = np.append(combined_dataset_X, signals_no_filter, axis=0)
            combined_dataset_y = np.append(combined_dataset_y, labels_no_filter, axis=0)
            combined_dataset_X = np.append(combined_dataset_X, signals_with_filter, axis=0)
            combined_dataset_y = np.append(combined_dataset_y, labels_with_filter, axis=0)

        datasets[f'{noise_percentage}% noise, {int(mix_ratio*100)}% mix, no filter'] = (signals_no_filter, labels_no_filter)
        datasets[f'{noise_percentage}% noise, {int(mix_ratio*100)}% mix, with LTE filter'] = (signals_with_filter, labels_with_filter)
for key, (signals, labels) in datasets.items():
    filename = f"datasets/dataset_{key.replace('%', 'pct').replace(' ', '_').replace(',', '').lower()}.npz"
    np.savez(filename, signals=signals, labels=labels)
        
# X, y = twod_dataset(datasets[list(datasets.keys())[1]])
generate = True
if generate:
    np.save('datasets/256size/combined_1d_x.npy', combined_dataset_X)
    np.save('datasets/256size/combined_1d_y.npy', combined_dataset_y)

    X = twod_dataset(combined_dataset_X)
    y = combined_dataset_y
    print(X.shape)
    print(y.shape)
    
    np.save('datasets/256size/combined_2d_x.npy', X)
    np.save('datasets/256size/combined_2d_y.npy', y)
else:
    X = np.load('combined_x.npy')
    y = np.load('combined_y.npy')

print(X.shape)