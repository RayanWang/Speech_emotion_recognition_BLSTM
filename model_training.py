from optparse import OptionParser
from dataset import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from random import randint

import numpy as np
import sys
import cPickle
import feature_extraction
import networks
import matplotlib.pyplot as plt


nb_classes = 7
nb_lstm_cells = 128
prefer_epochs = 39

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--database", dest="database", default="berlin")
    parser.add_option("-p", "--database_path", dest="path", default="")
    parser.add_option("-c", "--cross_validation", action="store_true", dest="cross_validation")
    parser.add_option("-l", "--load_data", action="store_true", dest="load_data")
    parser.add_option("-e", "--feature_extract", action="store_true", dest="feature_extract")
    parser.add_option("-s", "--speaker_independence", action="store_true", dest="speaker_independence")

    (options, args) = parser.parse_args(sys.argv)

    database = options.database
    path = options.path
    cross_validation = options.cross_validation
    load_data = options.load_data
    feature_extract = options.feature_extract
    speaker_independence = options.speaker_independence

    if load_data:
        print("Loading data from " + database + " database...")
        if database not in ('dafex', 'berlin'):
            sys.exit("Database not registered. Please create a method to read it")

        db = Dataset(path, database, decode=False)

        print("Saving " + database + " database info to file...")
        cPickle.dump(db, open(database + '_db.p', 'wb'))
    else:
        print("Getting data from " + database + " database...")
        db = cPickle.load(open(database + '_db.p', 'rb'))

    n_samples = len(db.targets)
    print("Number of database samples: " + str(n_samples))

    if feature_extract:
        print("Extracting features...")
        f_global = feature_extraction.train_extract(db.data, database=database)
    else:
        print("Getting features from files...")
        f_global = cPickle.load(open(database + '_features.p', 'rb'))

    y = np.array(db.targets)
    y = to_categorical(y, num_classes=nb_classes)

    if speaker_independence:
        k_folds = len(db.test_sets)
        splits = zip(db.train_sets, db.test_sets)
        print("Using speaker independence %s-fold cross validation" % k_folds)
    else:
        k_folds = 10
        sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=1)
        splits = sss.split(f_global, y)
        print("Using %s-fold cross validation by StratifiedShuffleSplit" % k_folds)

    # initialize the attention parameters
    nb_attention_param = nb_lstm_cells * 2
    init_value = 1.0 / nb_attention_param

    if cross_validation:
        cvscores = []

        for (train, test) in splits:
            # create network
            model = networks.create_softmax_la_network(input_shape=(f_global.shape[1], f_global.shape[2]))
            model.summary()

            # compile the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # initialize the attention parameters with all same values for training and validation
            u_train = np.full((f_global[train].shape[0], nb_attention_param,), init_value, dtype=np.float64)
            u_test = np.full((f_global[test].shape[0], nb_attention_param,), init_value, dtype=np.float64)

            # fit the model
            model.fit([u_train, f_global[train]], y[train], epochs=prefer_epochs, batch_size=128, verbose=1)

            # evaluate the model
            scores = model.evaluate([u_test, f_global[test]], y[test], verbose=1)

            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    else:
        callback_list = [
            EarlyStopping(
                monitor='acc',
                patience=5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath='blstm-{epoch:02d}-{val_loss:.2f}.hdf5',
                monitor='val_loss',
                save_best_only='True',
                verbose=1
            )
        ]

        # using the ith pair of train/test data
        i = randint(0, len(db.train_sets) - 1)
        train = db.train_sets[i]
        test = db.test_sets[i]
        print("Using the %sth train/test data..." % i)

        # create network
        model = networks.create_softmax_la_network(input_shape=(f_global.shape[1], f_global.shape[2]))
        model.summary()

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # initialize the attention parameters with all same values for training and validation
        u_train = np.full((f_global[train].shape[0], nb_attention_param,), init_value,
                          dtype=np.float64)
        u_test = np.full((f_global[test].shape[0], nb_attention_param,), init_value,
                         dtype=np.float64)

        # fit the model
        history = model.fit([u_train, f_global[train]], y[train], epochs=100, batch_size=128,
                            callbacks=callback_list, validation_data=([u_test, f_global[test]], y[test]),
                            verbose=1)

        history_dict = history.history
        epochs = range(1, len(history_dict['acc']) + 1)

        # show the training and validation accuracy
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']

        plt.plot(epochs, acc_values, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.clf()

        # show the training and validation loss
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
