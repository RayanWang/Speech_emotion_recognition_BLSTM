from keras.preprocessing.image import Iterator
from keras.preprocessing import sequence
from pyAudioAnalysis import audioFeatureExtraction
from utility import globalvars, audiosegment
from scipy import stats
from pydub import AudioSegment
from array import array
from struct import pack
from tqdm import tqdm

import numpy as np
import librosa
import os
import glob
import webrtcvad
import sys
import wave
import time
import soundfile as sf

try:
    import cPickle as pickle
except ImportError:
    import pickle


'''
Please be noted that the audio data generator doesn't tested yet. I'll test it later.
'''


def record_to_file(path, data, sample_width, sr=16000):
    """
    Records from the wav audio and outputs the resulting data to 'path'
    """
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(sr)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    """
    Average the volume out
    """
    max_value = max(abs(i) for i in snd_data)
    if max_value == 0:
        return snd_data
    maximum = 32767
    times = float(maximum) / max_value
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def extract_dataset(data, nb_samples, dataset, save=True):
    f_global = []

    i = 0
    for (x, Fs) in data:
        # 34D short-term feature
        f = audioFeatureExtraction.stFeatureExtraction(x, Fs, globalvars.frame_size * Fs, globalvars.step * Fs)

        # for pyAudioAnalysis which support python3
        if type(f) is tuple:
            f = f[0]

        # Harmonic ratio and pitch, 2D
        hr_pitch = audioFeatureExtraction.stFeatureSpeed(x, Fs, globalvars.frame_size * Fs, globalvars.step * Fs)
        f = np.append(f, hr_pitch.transpose(), axis=0)

        # Z-normalized
        f = stats.zscore(f, axis=0)

        f = f.transpose()

        f_global.append(f)

        sys.stdout.write("\033[F")
        i = i + 1
        print("Extracting features " + str(i) + '/' + str(nb_samples) + " from data set...")

    f_global = sequence.pad_sequences(f_global,
                                      maxlen=globalvars.max_len,
                                      dtype='float32',
                                      padding='post',
                                      value=globalvars.masking_value)

    if save:
        print("Saving features to file...")
        pickle.dump(f_global, open(dataset + '_features.p', 'wb'))

    return f_global


def extract(x, sr=16000):
    f_global = []

    # 34D short-term feature
    f = audioFeatureExtraction.stFeatureExtraction(x, sr, globalvars.frame_size * sr, globalvars.step * sr)

    # for pyAudioAnalysis which support python3
    if type(f) is tuple:
        f = f[0]

    # Harmonic ratio and pitch, 2D
    hr_pitch = audioFeatureExtraction.stFeatureSpeed(x, sr, globalvars.frame_size * sr, globalvars.step * sr)
    f = np.append(f, hr_pitch.transpose(), axis=0)

    # Z-normalized
    f = stats.zscore(f, axis=0)

    f = f.transpose()

    f_global.append(f)

    f_global = sequence.pad_sequences(f_global,
                                      maxlen=globalvars.max_len,
                                      dtype='float32',
                                      padding='post',
                                      value=globalvars.masking_value)

    return f_global


def griffinlim(spectrogram, n_iter=100, window='hann', n_fft=2048, hop_length=None, verbose=False):
    if hop_length is None:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)

    for _ in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))
        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return inverse


class AudioPreprocessing(object):

    def __init__(self, sr=16000, chunk_duration_ms=30, video_path='', out_path=''):
        self._sr = sr
        self._chunk_duration_ms = chunk_duration_ms
        self._chunk_size = int(sr * chunk_duration_ms / 1000)  # chunk to read in samples
        self._nb_window_chunks = int(400 / chunk_duration_ms)  # 400ms / 30ms frame
        self._nb_window_chunks_end = self._nb_window_chunks * 2
        self._vad = webrtcvad.Vad(mode=3)

        self._video_path = video_path
        self._out_path = out_path

    def extract_audio_track(self):
        for video in glob.glob(self._video_path + '*.mp4'):
            wav_filename = self._out_path + os.path.splitext(os.path.basename(video))[0] + '.wav'
            AudioSegment.from_file(video).export(wav_filename, format='wav')

    def sentence_slicing(self, filename, mode=0):
        print("Re-sampling...")
        seg = audiosegment.from_file(filename).resample(sample_rate_Hz=self._sr, sample_width=2, channels=1)

        print("Detecting voice...")

        got_sentence = False
        ended = False
        offset = self._chunk_duration_ms

        path = filename.split('/')[-1]
        path = self._out_path + '/' + path

        i = 1
        while not ended:
            triggered = False
            buffer_flags = [0] * self._nb_window_chunks
            buffer_index = 0

            buffer_flags_end = [0] * self._nb_window_chunks_end
            buffer_index_end = 0

            raw_data = array('h')
            index = 0
            start_point = 0
            start_time = time.time()

            start_timestamp = offset - self._chunk_duration_ms
            while not got_sentence and not ended:
                chunk = seg[(offset - self._chunk_duration_ms):offset].raw_data

                raw_data.extend(array('h', chunk))
                offset += self._chunk_duration_ms
                index += self._chunk_size
                time_used = time.time() - start_time

                active = self._vad.is_speech(chunk, self._sr)

                buffer_flags[buffer_index] = 1 if active else 0
                buffer_index += 1
                buffer_index %= self._nb_window_chunks

                buffer_flags_end[buffer_index_end] = 1 if active else 0
                buffer_index_end += 1
                buffer_index_end %= self._nb_window_chunks_end

                # start point detection
                if not triggered:
                    num_voiced = sum(buffer_flags)
                    if num_voiced > 0.8 * self._nb_window_chunks:
                        sys.stdout.write(' Start sentence ')
                        triggered = True
                        start_point = index - self._chunk_size * 20  # start point
                # end point detection
                else:
                    num_unvoiced = self._nb_window_chunks_end - sum(buffer_flags_end)
                    if num_unvoiced > 0.90 * self._nb_window_chunks_end or time_used > 10:
                        sys.stdout.write(' End sentence ')
                        triggered = False
                        got_sentence = True

                if offset >= len(seg):
                    sys.stdout.write(' File end ')
                    ended = True

                sys.stdout.flush()

            sys.stdout.write('\n')

            got_sentence = False

            print('Start point: %d' % start_point)

            # write to file
            raw_data.reverse()
            for _ in range(start_point):
                raw_data.pop()

            raw_data.reverse()
            raw_data = normalize(raw_data)

            print('Sentence length: %d bytes' % (len(raw_data) * 2))

            f, ext = os.path.splitext(path)
            if mode == 0:
                f = f + '_' + str(i) + ext
                i += 1
            else:
                # prediction
                start_timestamp += start_point / (self._sr / 1000)
                end_timestamp = start_timestamp + len(raw_data) / (self._sr / 1000)
                if (end_timestamp - start_timestamp) > 8000 or (end_timestamp - start_timestamp) < 3000:
                    continue

                f = f + '_' + str(start_timestamp) + '_' + str(end_timestamp) + ext

            record_to_file(f, raw_data, 2)


class AudioSplitter(object):

    def __init__(self, sr=16000, constrained=1.2):
        self._sr = sr
        self._constrained = constrained

    def split_vocal(self, y):
        S_full, phase = librosa.magphase(librosa.stft(y))

        # To avoid being biased by local continuity, we constrain similar frames to be
        # separated by at least 1.2 seconds.
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine',
                                               width=int(librosa.time_to_frames(self._constrained, sr=self._sr)))

        S_filter = np.minimum(S_full, S_filter)

        margin_v = 10
        power = 2

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        S_foreground = mask_v * S_full

        foreground = griffinlim(S_foreground)

        return foreground

    def split_vocal_to_wav(self, filename, fp_foreground, fp_background=None):
        print(filename.split('/')[-1])

        y, sr = librosa.load(filename, sr=self._sr)

        S_full, phase = librosa.magphase(librosa.stft(y))

        # To avoid being biased by local continuity, we constrain similar frames to be
        # separated by at least 1.2 seconds.
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine',
                                               width=int(librosa.time_to_frames(self._constrained, sr=self._sr)))

        S_filter = np.minimum(S_full, S_filter)

        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        foreground = griffinlim(S_foreground)
        fp_foreground += filename.split('/')[-1]
        sf.write(fp_foreground, foreground, sr, 'PCM_16')

        if fp_background is not None:
            background = griffinlim(S_background)
            fp_background += filename.split('/')[-1]
            sf.write(fp_background, background, sr, 'PCM_16')


class NumpyArrayIterator(Iterator):

    def __init__(self, x, y, audio_data_generator, sr=16000,
                 batch_size=32, shuffle=False, seed=None):
        if y is not None and len(x) != len(y):
            raise ValueError('`x` (audio tensor) and `y` (labels) '
                             'should have the same length. '
                             'Found: x.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = x
        self.y = y
        self.audio_data_generator = audio_data_generator
        self.sr = sr
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        for i, j in enumerate(index_array):
            x = self.x[j]

            # Augmentation
            if self.audio_data_generator.white_noise_:
                x = self.audio_data_generator.white_noise(x)
            if self.audio_data_generator.shift_:
                x = self.audio_data_generator.shift(x)
            if self.audio_data_generator.stretch_:
                x = self.audio_data_generator.stretch(x)

            # 34D short-term feature
            f = audioFeatureExtraction.stFeatureExtraction(x, self.sr, globalvars.frame_size * self.sr,
                                                           globalvars.step * self.sr)

            # Harmonic ratio and pitch, 2D
            hr_pitch = audioFeatureExtraction.stFeatureSpeed(x, self.sr, globalvars.frame_size * self.sr,
                                                             globalvars.step * self.sr)
            x = np.append(f, hr_pitch.transpose(), axis=0)

            # Z-normalized
            x = stats.zscore(x, axis=0)

            x = x.transpose()

            batch_x.append(x)

        batch_x = sequence.pad_sequences(batch_x,
                                         maxlen=globalvars.max_len,
                                         dtype='float32',
                                         padding='post',
                                         value=globalvars.masking_value)

        batch_u = np.full((len(index_array), globalvars.nb_attention_param,),
                          globalvars.attention_init_value,
                          dtype=np.float32)

        if self.y is None:
            return [batch_u, batch_x]
        batch_y = self.y[index_array]

        return [batch_u, batch_x], batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class AudioDataGenerator(object):

    def __init__(self,
                 sr=16000,
                 white_noise=False,
                 shift=False,
                 stretch=False,
                 wn_factor=0.005,
                 validation_split=0.0):
        self._sr = sr
        self.white_noise_ = white_noise
        self.shift_ = shift
        self.stretch_ = stretch
        self._wn_factor = wn_factor

        if validation_split and not 0 < validation_split < 1:
            raise ValueError('`validation_split` must be strictly between 0 and 1. '
                             ' Received arg: ', validation_split)
        self._validation_split = validation_split

    def flow(self, x, y=None, sr=16000, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayIterator(
            x, y, self,
            sr=sr,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)

    def white_noise(self, data):
        wn = np.random.randn(len(data))
        return data + self._wn_factor * wn

    def shift(self, data):
        shift = np.random.random_integers(self._sr // 10, self._sr // 2)
        return np.roll(data, shift)

    def stretch(self, data):
        rate = np.random.uniform(0.0, 2.0)
        input_length = self._sr
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), 'constant')

        return data
