from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence

import cPickle


maxlen = 1024
frame_size = 0.025  # 25 msec segments
step = 0.01     # 10 msec time step


def feature_normalization(f):
    f = f.transpose()
    f -= f.mean(axis=0)
    f /= f.std(axis=0)

    return f


def train_extract(data, database='berlin'):
    f_global = []

    for (x, Fs) in data:
        f = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size * Fs, step * Fs)

        # Feature normalization
        f = feature_normalization(f)

        f_global.append(f)

    f_global = sequence.pad_sequences(f_global, maxlen=maxlen, dtype='float64', padding='post', value=-100.0)

    print("Saving features to file...")
    cPickle.dump(f_global, open(database + '_features.p', 'wb'))

    return f_global


def predict_extract(signal, sr):
    f = audioFeatureExtraction.stFeatureExtraction(signal, sr, frame_size * sr, step * sr)

    # Feature normalization
    f = feature_normalization(f)

    return f
