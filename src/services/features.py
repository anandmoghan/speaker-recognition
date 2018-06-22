from resampy import resample
from scipy import signal

import numpy as np
import soundfile as sf


def compute_fft_spectrum(s, nfft, win_type='HAMMING', spectrum_type='pow', axis=-1):
    if s.ndim == 1:
        frame_len = s.size
    else:
        nframes, frame_len = s.shape
    win = window(frame_len, win_type, True)
    s = s * win
    fft_spectrum = np.fft.rfft(s, nfft, norm='ortho', axis=axis)
    fft_spectrum = fft_spectrum.real * fft_spectrum.real + fft_spectrum.imag * fft_spectrum.imag
    if spectrum_type == 'mag':
        fft_spectrum = np.sqrt(fft_spectrum)
    elif spectrum_type != 'pow':
        raise ValueError('Spectrum type {} not supported!'.format(spectrum_type))
    return fft_spectrum


def enframe(sig, frame_len, frame_inc):
    frames = rolling_window(sig, frame_len)
    if frames.ndim > 2:
        frames = np.rollaxis(frames, 1)
    return frames[::frame_inc]


def get_dct_matrix(n):
    m = range(n)
    x, y = np.meshgrid(m, m)
    dct = np.sqrt(2./n) * np.cos(np.pi * (2*x + 1) * y / (2*n))
    dct[0] /= np.sqrt(2)
    return dct


def hamming(n, periodic=False):
    total = n if periodic else n-1
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n+1)/total)
    return w[:-1]


def hanning(n, periodic=False):
    total = n if periodic else n-1
    w = np.sin(np.pi * np.arange(n+1)/total)**2
    return w[:-1]


def pre_emphasis(sig, mu=0.97):
    sig[1:] -= mu * sig[:-1]
    return sig


def remove_dc_n_dither(sig, fs):
    np.random.seed(7)
    if max(abs(sig)) <= 1:
        sig = sig * 32768  # assuming 16-bit
    if fs == 16e3:
        alpha = 0.99
    elif fs == 8e3:
        alpha = 0.999
    else:
        raise ValueError('Sampling frequency {} not supported'.format(fs))
    signal_len = sig.size
    sig = signal.lfilter([1, -1], [1, -alpha], sig)
    dither = np.random.rand(signal_len) + np.random.rand(signal_len) - 1
    sig_pow = max(sig.std(), 1e-20)
    return sig + 1.e-6 * sig_pow * dither


def rolling_window(a, window_len):
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1, window_len)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window(frame_len, win_type='RECT', periodic=False):
    if win_type == 'RECT':
        return np.arange(frame_len)
    elif win_type == 'HAMMING':
        return hamming(frame_len, periodic)
    elif win_type == 'HANNING':
        return hanning(frame_len, periodic)
    else:
        raise ValueError('Window {} not supported'.format(win_type))


class MFCC:
    def __init__(self, fs=8000, nfft=512, frame_len=25, frame_inc=10, pre_emph_coef=0.97, nchannels=24, fl=100, fh=4000,
                 nceps=20, filter_shape=None, win_type='HAMMING', spectrum_type='mag'):
        self.fs = fs
        self.nfft = nfft
        self.frame_len = int(frame_len * fs / 1000)
        self.frame_inc = int(frame_inc * fs / 1000)
        self.pre_emph_coef = pre_emph_coef
        self.nchannels = nchannels
        self.fl = fl
        self.fh = min(fh, int(self.fs / 2))
        self.nceps = nceps
        self.win_type = win_type
        self.spectrum_type = spectrum_type
        self.melb = self.mel_bank(filter_shape)
        self.dctmat = get_dct_matrix(nchannels)[:nceps]

    def mel_bank(self, filter_shape=None):
        mel_fl = MFCC.hz2mel(self.fl)
        mel_fh = MFCC.hz2mel(self.fh)
        edges = MFCC.mel2hz(np.linspace(mel_fl, mel_fh, self.nchannels + 2))
        fft_bins = np.round(edges / self.fs * self.nfft).astype(int)
        diff1 = fft_bins[1:-1] - fft_bins[0:-2]
        diff2 = fft_bins[2:] - fft_bins[1:-1]
        m = np.zeros((int(self.nfft / 2 + 1), self.nchannels))
        for d in range(self.nchannels):
            m[fft_bins[d]:fft_bins[d + 1], d] = np.arange(diff1[d]) / diff1[d]
            m[fft_bins[d + 1]:fft_bins[d + 2], d] = np.arange(diff2[d], 0, -1) / diff2[d]

        if filter_shape == 'HAMMING':
            m = 0.5 - 0.46 * np.cos(np.pi * m)
        elif filter_shape == 'HANNING':
            m = 0.5 - 0.5 * np.cos(np.pi * m)
        return m

    def extract(self, speech):
        s = remove_dc_n_dither(speech, self.fs)
        s = pre_emphasis(s, self.pre_emph_coef)
        s = enframe(s, self.frame_len, self.frame_inc)
        fft_spectrum = compute_fft_spectrum(s, self.nfft, self.win_type, self.spectrum_type)
        log_e = np.log(fft_spectrum.dot(self.melb) + np.finfo(float).eps)
        mfcc = log_e.dot(self.dctmat.T).T
        return mfcc

    def extract_files(self, file_list):
        mfcc = []
        for file_name in file_list:
            speech, sr = sf.read(file_name)
            if sr > self.fs:
                speech = resample(speech, sr, self.fs)
            mfcc.append([self.extract(speech)])
        return mfcc

    @staticmethod
    def hz2mel(f):
        return np.log(1 + f/700.) / np.log(1 + 1000./700) / 1000

    @staticmethod
    def mel2hz(mel):
        return 700 * (np.exp(mel * np.log(1 + 1000./700) / 1000) - 1)
