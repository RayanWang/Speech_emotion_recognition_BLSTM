from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence
from scipy import stats

import numpy as np
import cPickle
import sys
import globalvars


def feature_extract(data, nb_samples, dataset='berlin'):
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


def get_confusion_matrix_one_hot(model_results, truth):
    '''
    model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:, actual_class] == 1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class == predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix) == len(truth)
    assert np.sum(confusion_matrix) == np.sum(truth)

    return confusion_matrix
