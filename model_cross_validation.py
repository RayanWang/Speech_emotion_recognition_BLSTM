from optparse import OptionParser
from dataset import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical

import numpy as np
import sys
import cPickle
import feature_extraction
import globalvars
import networks


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset', dest='dataset', default='dafex')
    parser.add_option('-p', '--dataset_path', dest='path', default='')
    parser.add_option('-l', '--load_data', action='store_true', dest='load_data')
    parser.add_option('-e', '--feature_extract', action='store_true', dest='feature_extract')
    parser.add_option('-s', '--speaker_independence', action='store_true', dest='speaker_independence')

    (options, args) = parser.parse_args(sys.argv)

    dataset = options.dataset
    path = options.path
    load_data = options.load_data
    feature_extract = options.feature_extract
    speaker_independence = options.speaker_independence

    if load_data:
        print("Loading data from " + dataset + " data set...")
        if dataset not in ('dafex', 'berlin'):
            sys.exit("Dataset not registered. Please create a method to read it")

        db = Dataset(path, dataset, decode=False)

        print("Dumping " + dataset + " data set to file...")
        cPickle.dump(db, open(dataset + '_db.p', 'wb'))
    else:
        print("Getting data from " + dataset + " data set...")
        db = cPickle.load(open(dataset + '_db.p', 'rb'))

    nb_samples = len(db.targets)
    print("Number of samples: " + str(nb_samples))

    if feature_extract:
        f_global = feature_extraction.training_extract(db.data, nb_samples=nb_samples, dataset=dataset)
    else:
        print("Getting features from files...")
        f_global = cPickle.load(open(dataset + '_features.p', 'rb'))

    y = np.array(db.targets)
    y = to_categorical(y, num_classes=globalvars.nb_classes)

    if speaker_independence:
        k_folds = len(db.test_sets)
        splits = zip(db.train_sets, db.test_sets)
        print("Using speaker independence %s-fold cross validation" % k_folds)
    else:
        k_folds = 10
        sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=1)
        splits = sss.split(f_global, y)
        print("Using %s-fold cross validation by StratifiedShuffleSplit" % k_folds)

    cvscores = []

    for (train, test) in splits:
        # initialize the attention parameters with all same values for training and validation
        u_train = np.full((f_global[train].shape[0], globalvars.nb_attention_param,),
                          globalvars.attention_init_value, dtype=np.float64)
        u_test = np.full((f_global[test].shape[0], globalvars.nb_attention_param),
                         globalvars.attention_init_value, dtype=np.float64)

        # create network
        model = networks.create_softmax_la_network(input_shape=(f_global.shape[1], f_global.shape[2]))

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit the model
        model.fit([u_train, f_global[train]], y[train], epochs=100, batch_size=128, verbose=1)

        # evaluate the model
        scores = model.evaluate([u_test, f_global[test]], y[test], verbose=1)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
