from optparse import OptionParser
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as k
from keras.utils import to_categorical
from utility import networks, globalvars, metrics_util
from utility.audio import AudioDataGenerator
from dataset import Dataset

import numpy as np
import sys
import cPickle
import math


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset', dest='dataset', default='berlin')
    parser.add_option('-p', '--dataset_path', dest='path', default='')
    parser.add_option('-l', '--load_data', action='store_true', dest='load_data')
    parser.add_option('-e', '--feature_extract', action='store_true', dest='feature_extract')
    parser.add_option('-c', '--nb_classes', dest='nb_classes', type='int', default=6)
    parser.add_option('-s', '--speaker_independence', action='store_true', dest='speaker_independence')

    (options, args) = parser.parse_args(sys.argv)

    dataset = options.dataset
    path = options.path
    load_data = options.load_data
    feature_extract = options.feature_extract
    nb_classes = options.nb_classes
    speaker_independence = options.speaker_independence

    globalvars.nb_classes = nb_classes

    batch_size = 32

    if load_data:
        print("Loading data from " + dataset + " data set...")
        if dataset not in ('dafex', 'berlin'):
            sys.exit("Dataset not registered. Please create a method to read it")

        ds = Dataset(path, dataset, decode=False)

        print("Dumping " + dataset + " data set to file...")
        cPickle.dump(ds, open(dataset + '_db.p', 'wb'))
    else:
        print("Loading data from " + dataset + " data set...")
        ds = cPickle.load(open(dataset + '_db.p', 'rb'))

    nb_samples = len(ds.targets)
    print("Number of samples: " + str(nb_samples))

    x = np.array([x for x, _ in ds.data])
    y = np.array(ds.targets)
    y = to_categorical(y)

    if speaker_independence:
        k_folds = len(ds.test_sets)
        splits = zip(ds.train_sets, ds.test_sets)
        print("Using speaker independence %s-fold cross validation" % k_folds)
    else:
        k_folds = 5
        sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=1)
        splits = sss.split(x, y)
        print("Using %s-fold cross validation by StratifiedShuffleSplit" % k_folds)

    cvscores = []
    for (train, test) in splits:
        # create network
        model = networks.create_softmax_la_network(input_shape=(globalvars.max_len, globalvars.nb_features),
                                                   nb_classes=nb_classes)

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        file_path = 'weights.h5'
        callback_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                mode='auto'
            ),
            ModelCheckpoint(
                filepath=file_path,
                monitor='val_acc',
                save_best_only='True',
                verbose=1,
                mode='max'
            )
        ]

        train_gen = AudioDataGenerator(sr=16000,
                                       white_noise=True,
                                       shift=True,
                                       stretch=True)
        validate_gen = AudioDataGenerator(sr=16000,
                                          white_noise=True)

        steps = math.ceil(len(x[train]) / batch_size)
        validation_steps = math.ceil(len(x[test]) / batch_size)

        # fit the model
        hist = model.fit_generator(train_gen.flow(x=x[train], y=y[train]),
                                   steps_per_epoch=steps,
                                   epochs=200,
                                   callbacks=callback_list,
                                   validation_data=validate_gen.flow(x=x[test], y=y[test]),
                                   validation_steps=validation_steps,
                                   use_multiprocessing=True)

        # evaluate the best model in ith fold
        best_model = load_model(file_path)

        print("Getting the confusion matrix on whole set...")
        predict_gen = AudioDataGenerator(sr=16000,
                                         white_noise=True)
        predictions = best_model.predict_generator(predict_gen.flow(x=x, y=y),
                                                   use_multiprocessing=True,
                                                   verbose=1)
        confusion_matrix = metrics_util.get_confusion_matrix_one_hot(predictions, y)
        print(confusion_matrix)

        break

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    k.clear_session()
