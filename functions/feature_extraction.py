from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence

import cPickle
import sys
import globalvars


def feature_normalization(f):
    f = f.transpose()
    f -= f.mean(axis=0)
    f /= f.std(axis=0)

    return f


def training_extract(data, nb_samples, dataset='berlin'):
    f_global = []

    i = 0
    for (x, Fs) in data:
        f = audioFeatureExtraction.stFeatureExtraction(x, Fs, globalvars.frame_size * Fs,
                                                       globalvars.step * Fs)

        # Feature normalization
        f = feature_normalization(f)

        f_global.append(f)

        sys.stdout.write("\033[F")
        i = i + 1
        print "Extracting features " + str(i) + '/' + str(nb_samples) + " from data set..."

    f_global = sequence.pad_sequences(f_global, maxlen=globalvars.max_len, dtype='float64',
                                      padding='post', value=-100.0)

    print("Saving features to file...")
    cPickle.dump(f_global, open(dataset + '_features.p', 'wb'))

    return f_global


def prediction_extract(signal, sr):
    f = audioFeatureExtraction.stFeatureExtraction(signal, sr, globalvars.frame_size * sr,
                                                   globalvars.step * sr)

    # Feature normalization
    f = feature_normalization(f)

    return f
