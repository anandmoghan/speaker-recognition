from subprocess import Popen, PIPE
from scipy import signal
from os.path import join as join_path

import numpy as np
import logging
import resampy

from constants.app_constants import MFCC_DIR, SAD_DIR, SAD_LIST_FILE
from services.common import load_array, make_directory, run_parallel, save_array


class MFCC:
    def __init__(self, fs=8000, n_fft=512, frame_len_ms=25, frame_inc_ms=10,
                 pre_emph_coef=0.97, n_channels=24, fl=100, fh=4000, n_ceps=20,
                 filter_shape='', win_type='hamm', spectrum_type='mag', compression='log', save_loc='../save'):
        self.fs = fs
        self.nfft = n_fft
        self.frame_len = int(frame_len_ms * fs / 1000)
        self.frame_inc = int(frame_inc_ms * fs / 1000)
        self.pre_emph_coef = pre_emph_coef
        self.n_channels = n_channels
        self.fl = fl
        self.fh = min(fh, int(self.fs/2))
        self.n_ceps = n_ceps
        self.win_type = win_type
        self.spectrum_type = spectrum_type
        self.compression = compression
        self.mel_bank_coef = self.mel_bank(filter_shape)
        self.dct_matrix = get_dct_matrix(n_channels)[:n_ceps]
        self.save_loc = join_path(save_loc, MFCC_DIR)
        self.sad_loc = join_path(save_loc, SAD_DIR)
        make_directory(self.save_loc)

    def mel_bank(self, filter_shape=None):
        mel_fl = MFCC.hz2mel(self.fl)
        mel_fh = MFCC.hz2mel(self.fh)
        edges = MFCC.mel2hz(np.linspace(mel_fl, mel_fh, self.n_channels + 2))
        fft_bins = np.round(edges / self.fs * self.nfft).astype(int)
        diff1 = fft_bins[1:-1] - fft_bins[0:-2]
        diff2 = fft_bins[2:] - fft_bins[1:-1]
        mel_coef = np.zeros((int(self.nfft/2 + 1), self.n_channels))
        for d in range(self.n_channels):
            mel_coef[fft_bins[d]:fft_bins[d+1], d] = np.arange(diff1[d]) / diff1[d]
            mel_coef[fft_bins[d+1]:fft_bins[d+2], d] = np.arange(diff2[d], 0, -1) / diff2[d]

        if filter_shape == 'hamm':
            mel_coef = 0.5 - 0.46 * np.cos(np.pi * mel_coef)
        elif filter_shape == 'hann':
            mel_coef = 0.5 - 0.5 * np.cos(np.pi * mel_coef)
        return mel_coef

    def extract(self, speech):
        processed_speech = rm_dc_n_dither(speech, self.fs)
        processed_speech = pre_emphasis(processed_speech, self.pre_emph_coef)
        processed_speech = enframe(processed_speech, self.frame_len, self.frame_inc)
        fft_spectrum = compute_fft_spectrum(processed_speech, self.nfft, self.win_type, self.spectrum_type)
        if self.compression == 'log':
            log_e = np.log(fft_spectrum.dot(self.mel_bank_coef))
        elif self.compression == 'plaw':
            log_e = (fft_spectrum.dot(self.mel_bank_coef)) ** (1. / 15)
        else:
            raise ValueError('Compression type {} not supported!'.format(self.compression))
        mfc = log_e.dot(self.dct_matrix.T).T
        return mfc

    def extract_with_sad_and_cmvn(self, speech, sad_loc):
        features = self.extract(speech)
        sad = read_3col_sad(sad_loc, features.shape[1])
        features = rasta_filter(features)
        features = np.r_[features[:, sad]]
        features = cmvn(features)
        features = window_cmvn(features, window_len=301, var_norm=False)
        return features

    def extract_sph_file_with_sad_and_cmvn(self, args, save=True):
        speech, sr = read_sph_audio(args[1], args[2])
        if sr > self.fs:
            speech = resample(speech, sr, self.fs)
        logging.info('Extracting MFCC: {}'.format(args[0]))
        sad_loc = join_path(self.sad_loc, args[0] + '.sad')
        mfcc = self.extract_with_sad_and_cmvn(speech, sad_loc)
        if save:
            file_name = join_path(self.save_loc, str(args[0]) + '.npy')
            save_array(file_name, mfcc)
            logging.info('Saved {} as {}'.format(args[0], file_name))
            return mfcc.shape[1]
        return mfcc

    def extract_sph_files(self, args_list):
        mfcc = []
        for args in args_list:
            speech, sr = read_sph_audio(args[1], args[2])
            if sr > self.fs:
                speech = resample(speech, sr, self.fs)
                logging.info('Extracting MFCC for: {}'.format(args[0]))
            mfcc.append([self.extract(speech)])
        return mfcc

    def extract_sph_files_with_sad_and_cmvn(self, args_list):
        mfcc = []
        for args in args_list:
            mfcc.append([self.extract_sph_file_with_sad_and_cmvn(args, save=False)])
        return mfcc

    @staticmethod
    def hz2mel(f):
        mel_1000 = np.log(1 + 1000./700) / 1000
        return np.log(1 + f/700.) / mel_1000

    @staticmethod
    def mel2hz(mel):
        mel_1000 = np.log(1 + 1000./700) / 1000
        return 700 * (np.exp(mel * mel_1000) - 1)


def append_deltas(frames, delta_window_len=5, double_delta_window_len=5):
    delta_frames = deltas(frames, delta_window_len)
    frames = np.r_[frames, delta_frames]
    if double_delta_window_len > 0:
        frames = np.r_[frames, deltas(delta_frames, double_delta_window_len)]
    return frames


# def append_shifted_deltas(x, N=7, d=1, P=3, k=7):
#     if d < 1:
#         raise ValueError('d should be an integer >= 1')
#     n_obs = x.shape[1]
#     x = x[:N]
#     w = 2 * d + 1
#     dx = deltas(x, w)
#     sdc = np.empty((k*N, n_obs))
#     sdc[:] = np.tile(dx[:, -1], k).reshape(k*N, 1)
#     for ix in range(k):
#         if ix*P > n_obs:
#             break
#         sdc[ix*N:(ix+1)*N, :n_obs-ix*P] = dx[:, ix*P:n_obs]
#     return sdc


def compute_fft_spectrum(speech, nfft, win_type='hamm', spectrum_type='pow', axis=-1):
    if speech.ndim == 1:
        frame_len = speech.size
    else:
        nframes, frame_len = speech.shape
    window_func = window(frame_len, win_type, True)
    speech = speech * window_func
    fft_spectrum = np.fft.rfft(speech, nfft, norm='ortho', axis=axis)
    fft_spectrum = fft_spectrum.real * fft_spectrum.real + fft_spectrum.imag * fft_spectrum.imag
    if spectrum_type == 'mag':
        fft_spectrum = np.sqrt(fft_spectrum)
    elif spectrum_type != 'pow':
        raise ValueError('Spectrum type {} not supported!'.format(spectrum_type))
    return fft_spectrum


def cmvn(x, var_norm=True):
    y = x - x.mean(1, keepdims=True)
    if var_norm:
        y /= (x.std(1, keepdims=True) + 1e-20)
    return y


def deltas(x, window_len=5):
    if window_len < 3 or (window_len & 1) != 1:
        raise ValueError('Window length should be an odd integer >= 3')
    h_len = int(window_len / 2.)
    win = np.arange(h_len, -(h_len+1), -1)
    win = win / np.sum(win * win)
    xx = np.c_[np.tile(x[:, 0][:, np.newaxis], h_len), x, np.tile(x[:, -1][:, np.newaxis], h_len)]
    d = signal.lfilter(win, 1, xx)
    return d[:, 2*h_len:]


def enframe(sig, frame_len, frame_inc):
    frames = rolling_window(sig, frame_len)
    if frames.ndim > 2:
        frames = np.rollaxis(frames, 1)
    return frames[::frame_inc]


def generate_sad_list(save_loc, args_list, append=False):
    sad_list_file = join_path(save_loc, SAD_LIST_FILE)
    with open(sad_list_file, 'a' if append else 'w') as f:
        for args in args_list:
            f.write('{}, {}, {}/{}/{}.sad\n'.format(args[1], ('a' if args[2] == '1' else 'b'), save_loc, SAD_DIR,
                                                    args[0]))


def get_dct_matrix(n):
    m = range(n)
    x, y = np.meshgrid(m, m)
    dct_matrix = np.sqrt(2./n) * np.cos(np.pi * (2*x + 1) * y / (2*n))
    dct_matrix[0] /= np.sqrt(2)
    return dct_matrix


def get_frame(file_loc):
    return load_array(file_loc).shape[1]


def get_mfcc_frames(save_loc, args):
    mfcc_loc = join_path(save_loc, MFCC_DIR)
    file_loc = []
    for a in args:
        file_loc.append(join_path(mfcc_loc, a + '.npy'))
    return np.array(run_parallel(get_frame, file_loc)).reshape([-1, 1])


def hamming(n, periodic=False):
    window_len = n if periodic else n-1
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n+1) / window_len)
    return w[:-1]


def hanning(n, periodic=False):
    window_len = n if periodic else n-1
    w = np.sin(np.pi * np.arange(n+1) / window_len)**2
    return w[:-1]


def load_feature(file_name):
    return load_array(file_name)


def l2norm(x, axis=0):
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=True))


def pre_emphasis(sig, mu=0.97):
    sig[1:] -= mu * sig[:-1]
    return sig


def rasta_filter(x):
    n_dim, n_obs = x.shape
    n = np.arange(-2, 3)
    n = -n / np.sum(n * n)
    d = [1, -0.94]
    y = np.zeros((n_dim, 4))
    z = np.zeros((n_dim, 4))
    zi = [0., 0., 0., 0.]
    for ix in range(n_dim):
        y[ix, :], z[ix, :] = signal.lfilter(n, 1, x[ix, :4], zi=zi, axis=-1)
    y = np.zeros((n_dim, n_obs))
    for ix in range(n_dim):
        y[ix, 4:] = signal.lfilter(n, d, x[ix, 4:], zi=z[ix, :], axis=-1)[0]
    return y


def read_3col_sad(filename, nobs):
    sad = np.zeros((nobs,), dtype=np.bool)
    try:
        with open(filename, 'r') as fid:
            lines = fid.read().splitlines()
        for line in lines:
            fields = line.split()
            be = int(100 * float(fields[1]))
            en = min(int(100 * float(fields[2])), nobs)
            sad[be:en] = True
    except FileNotFoundError:
        pass
    except ValueError:
        pass
    if sad.sum() == 0:
        sad = np.ones((nobs,), dtype=np.bool)
    return sad


def read_sph_audio(filename, ch=1):
    cmd = "sph2pipe -f wav -p -c {} {}".format(ch, filename)
    p = Popen(cmd, stdout=PIPE, shell=True)
    output, error_output = p.communicate()
    sample_rate = np.frombuffer(output, dtype='uint32', count=1, offset=24)[0]
    data = np.frombuffer(output, dtype=np.int16, count=-1, offset=44).astype('f8')
    data /= 2**15  # assuming 16-bit
    return data, sample_rate


def resample(speech, sr_old, sr_new):
    if sr_old != sr_new:
            speech = resampy.resample(speech, sr_old, sr_new)
    return speech


def rm_dc_n_dither(speech, fs):
    np.random.seed(7)  # for repeatability
    if max(abs(speech)) <= 1:
        speech = speech * 32768  # assuming 16-bit
    if fs == 16e3:
        alpha = 0.99
    elif fs == 8e3:
        alpha = 0.999
    else:
        raise ValueError('Sampling frequency {} not supported'.format(fs))
    speech_len = speech.size
    speech = signal.lfilter([1, -1], [1, -alpha], speech)
    dither = np.random.rand(speech_len) + np.random.rand(speech_len) - 1
    sig_pow = max(speech.std(), 1e-20)
    return speech + 1.e-6 * sig_pow * dither


def rolling_window(a, window_len):
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1, window_len)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def splice_feats(x, window_len=9):
    if window_len < 3 or ((window_len & 1) != 1):
        raise ValueError('Window length should be an odd integer >= 3')
    h_len = int(window_len / 2.)
    n_dim, n_obs = x.shape
    xx = np.c_[np.tile(x[:, 0][:, np.newaxis], h_len), x,
               np.tile(x[:, -1][:, np.newaxis], h_len)]
    y = np.empty((window_len * n_dim, n_obs), dtype=x.dtype)
    for ix in range(window_len):
        y[ix*n_dim:(ix+1)*n_dim, :] = xx[:, ix:ix+n_obs]
    return y


def window(frame_len, win_type='rect', periodic=False):
    if win_type == 'rect':
        return np.arange(frame_len)
    elif win_type == 'hamm':
        return hamming(frame_len, periodic)
    elif win_type == 'hann':
        return hanning(frame_len, periodic)
    return None


def window_cmvn(x, window_len=301, var_norm=True):
    if window_len < 3 or (window_len & 1) != 1:
        raise ValueError('Window length should be an odd integer >= 3')
    n_dim, n_obs = x.shape
    if n_obs < window_len:
        return cmvn(x, var_norm)
    h_len = int((window_len - 1) / 2)
    y = np.zeros((n_dim, n_obs), dtype=x.dtype)
    y[:, :h_len] = x[:, :h_len] - x[:, :window_len].mean(1, keepdims=True)
    for ix in range(h_len, n_obs-h_len):
        y[:, ix] = x[:, ix] - x[:, ix-h_len:ix+h_len+1].mean(1)
    y[:, n_obs-h_len:n_obs] = x[:, n_obs-h_len:n_obs] - x[:, n_obs - window_len:].mean(1, keepdims=True)
    if var_norm:
        y[:, :h_len] /= (x[:, :window_len].std(1, keepdims=True) + 1e-20)
        for ix in range(h_len, n_obs-h_len):
            y[:, ix] /= (x[:, ix-h_len:ix+h_len+1].std(1) + 1e-20)
        y[:, n_obs-h_len:n_obs] /= (x[:, n_obs - window_len:].std(1, keepdims=True) + 1e-20)
    return y
