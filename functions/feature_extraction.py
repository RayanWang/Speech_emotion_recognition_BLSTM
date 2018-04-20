from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence
from scipy import stats

import numpy as np
import cPickle
import sys
import globalvars


def training_extract(data, nb_samples, dataset='berlin'):
    f_global = []

    i = 0
    for (x, Fs) in data:
        # 34D short-term feature
        f = audioFeatureExtraction.stFeatureExtraction(x, Fs, globalvars.frame_size * Fs, globalvars.step * Fs)

        # Harmonic ratio and pitch, 2D
        hr_pitch = audioFeatureExtraction.stFeatureSpeed(x, Fs, globalvars.frame_size * Fs, globalvars.step * Fs)
        f = np.append(f, hr_pitch.transpose(), axis=0)

        # Z-normalized
        f = stats.zscore(f, axis=0)

        f = f.transpose()

        f_global.append(f)

        sys.stdout.write("\033[F")
        i = i + 1
        print "Extracting features " + str(i) + '/' + str(nb_samples) + " from data set..."

    f_global = sequence.pad_sequences(f_global, maxlen=globalvars.max_len, dtype='float64',
                                      padding='post', value=-100.0)

    print("Saving features to file...")
    cPickle.dump(f_global, open(dataset + '_features.p', 'wb'))

    return f_global
