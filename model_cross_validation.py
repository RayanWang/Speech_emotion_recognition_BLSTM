from optparse import OptionParser
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from utility import networks, metrics_util, globalvars
from utility.audio import extract_dataset
from dataset import Dataset

import numpy as np
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset', dest='dataset', default='berlin')
    parser.add_option('-p', '--dataset_path', dest='path', default='')
    parser.add_option('-l', '--load_data', action='store_true', dest='load_data')
    parser.add_option('-e', '--feature_extract', action='store_true', dest='feature_extract')
    parser.add_option('-c', '--nb_classes', dest='nb_classes', type='int', default=7)
    parser.add_option('-s', '--speaker_independence', action='store_true', dest='speaker_independence')

    (options, args) = parser.parse_args(sys.argv)

    dataset = options.dataset
    path = options.path
    load_data = options.load_data
    feature_extract = options.feature_extract
    nb_classes = options.nb_classes
    speaker_independence = options.speaker_independence

    globalvars.nb_classes = nb_classes

    if load_data:
        print("Loading data from " + dataset + " data set...")
        if dataset not in ('dafex', 'berlin'):
            sys.exit("Dataset not registered. Please create a method to read it")

        ds = Dataset(path, dataset, decode=False)

        print("Dumping " + dataset + " data set to file...")
        pickle.dump(ds, open(dataset + '_db.p', 'wb'))
    else:
        print("Loading data from " + dataset + " data set...")
        ds = pickle.load(open(dataset + '_db.p', 'rb'))

    nb_samples = len(ds.targets)
    print("Number of samples: " + str(nb_samples))

    if feature_extract:
        f_global = extract_dataset(ds.data, nb_samples=nb_samples, dataset=dataset)
    else:
        print("Loading features from file...")
        f_global = pickle.load(open(dataset + '_features.p', 'rb'))

    y = np.array(ds.targets)
    y = to_categorical(y)

    if speaker_independence:
        k_folds = len(ds.test_sets)
        splits = zip(ds.train_sets, ds.test_sets)
        print("Using speaker independence %s-fold cross validation" % k_folds)
    else:
        k_folds = 10
        sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=1)
        splits = sss.split(f_global, y)
        print("Using %s-fold cross validation by StratifiedShuffleSplit" % k_folds)

    cvscores = []

    i = 1
    for (train, test) in splits:
        # initialize the attention parameters with all same values for training and validation
        u_train = np.full((len(train), globalvars.nb_attention_param),
                          globalvars.attention_init_value, dtype=np.float32)
        u_test = np.full((len(test), globalvars.nb_attention_param),
                         globalvars.attention_init_value, dtype=np.float32)

        # create network
        model = networks.create_softmax_la_network(input_shape=(globalvars.max_len, globalvars.nb_features),
                                                   nb_classes=nb_classes)

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        file_path = 'weights_' + str(i) + '_fold' + '.h5'
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
            ),
            TensorBoard(
                log_dir='./Graph',
                histogram_freq=0,
                write_graph=True,
                write_images=True
            )
        ]

        # fit the model
        hist = model.fit([u_train, f_global[train]],
                         y[train],
                         epochs=200,
                         batch_size=128,
                         verbose=2,
                         callbacks=callback_list,
                         validation_data=([u_test, f_global[test]], y[test]))

        # evaluate the best model in ith fold
        best_model = load_model(file_path)

        print("Evaluating on test set...")
        scores = best_model.evaluate([u_test, f_global[test]], y[test], batch_size=128, verbose=1)
        print("The highest %s in %dth fold is %.2f%%" % (model.metrics_names[1], i, scores[1] * 100))

        cvscores.append(scores[1] * 100)

        print("Getting the confusion matrix on whole set...")
        u = np.full((f_global.shape[0], globalvars.nb_attention_param),
                    globalvars.attention_init_value, dtype=np.float32)
        predictions = best_model.predict([u, f_global])
        confusion_matrix = metrics_util.get_confusion_matrix_one_hot(predictions, y)
        print(confusion_matrix)

        i += 1

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
