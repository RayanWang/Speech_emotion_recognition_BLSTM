from optparse import OptionParser

from utility import functions, globalvars
from dataset import Dataset

from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.layers.merge import dot
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from hyperas import optim
from hyperopt import Trials, tpe
from hyperopt import STATUS_OK
from hyperas.distributions import choice

import sys
import cPickle
import numpy as np


def get_data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    print("Loading data and features...")
    db = cPickle.load(open(globalvars.dataset + '_db.p', 'rb'))
    f_global = cPickle.load(open(globalvars.dataset + '_features.p', 'rb'))

    nb_samples = len(db.targets)
    print("Number of samples: " + str(nb_samples))

    y = np.array(db.targets)
    y = to_categorical(y, num_classes=globalvars.nb_classes)

    train = db.train_sets[0]
    test = db.test_sets[0]

    u_train = np.full((f_global[train].shape[0], globalvars.nb_attention_param),
                      globalvars.attention_init_value, dtype=np.float64)
    x_train = f_global[train]
    y_train = y[train]

    u_test = np.full((f_global[test].shape[0], globalvars.nb_attention_param),
                     globalvars.attention_init_value, dtype=np.float64)
    x_test = f_global[test]
    y_test = y[test]

    return u_train, x_train, y_train, u_test, x_test, y_test


def create_model(u_train, x_train, y_train, u_test, x_test, y_test):
    # Logistic regression for learning the attention parameters with a standalone feature as input
    input_attention = Input(shape=(globalvars.nb_attention_param,))
    u = Dense(globalvars.nb_attention_param, activation='softmax')(input_attention)

    # Bi-directional Long Short-Term Memory for learning the temporal aggregation
    # Input shape: (time_steps, features,)
    input_feature = Input(shape=(globalvars.max_len, globalvars.nb_features))
    x = Masking(mask_value=-100.0)(input_feature)

    x = Dense(globalvars.nb_hidden_units, activation='relu')(x)
    x = Dropout(globalvars.dropout_rate)(x)

    x = Dense(globalvars.nb_hidden_units, activation='relu')(x)
    x = Dropout(globalvars.dropout_rate)(x)

    y = Bidirectional(LSTM(globalvars.nb_lstm_cells, return_sequences=True,
                           dropout=globalvars.dropout_rate))(x)

    # To compute the final weights for the frames which sum to unity
    alpha = dot([u, y], axes=-1)
    alpha = Activation('softmax')(alpha)

    # Weighted pooling to get the utterance-level representation
    z = dot([alpha, y], axes=1)

    # Get posterior probability for each emotional class
    output = Dense(globalvars.nb_classes, activation='softmax')(z)

    model = Model(inputs=[input_attention, input_feature], outputs=output)

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    choice_val = {{choice(['adam', 'rmsprop'])}}
    if choice_val == 'adam':
        optimizer = optimizers.Adam()
    elif choice_val == 'rmsprop':
        optimizer = optimizers.RMSprop()
    else:
        # optimizer = sgd
        optimizer = optimizers.RMSprop()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    globalvars.globalVar += 1

    file_path = "weights_blstm_hyperas_" + str(globalvars.globalVar) + ".hdf5"
    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            filepath=file_path,
            monitor='val_acc',
            save_best_only='True',
            verbose=1,
            mode='max'
        )
    ]

    hist = model.fit([u_train, x_train], y_train, batch_size=128, epochs={{choice([250, 300])}}, verbose=2,
                     callbacks=callback_list, validation_data=([u_test, x_test], y_test))
    h = hist.history
    acc = np.asarray(h['acc'])
    loss = np.asarray(h['loss'])
    val_loss = np.asarray(h['val_loss'])
    val_acc = np.asarray(h['val_acc'])

    acc_and_loss = np.column_stack((acc, loss, val_acc, val_loss))
    save_file_blstm = "blstm_run_" + str(globalvars.globalVar) + ".txt"
    with open(save_file_blstm, 'w') as f:
        np.savetxt(save_file_blstm, acc_and_loss)

    score, accuracy = model.evaluate([u_test, x_test], y_test, verbose=1)
    print("Final validation accuracy: %s" % accuracy)

    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset', dest='dataset', default='berlin')
    parser.add_option('-p', '--dataset_path', dest='path', default='')
    parser.add_option('-l', '--load_data', action='store_true', dest='load_data')
    parser.add_option("-e", "--feature_extract", action="store_true", dest="feature_extract")

    (options, args) = parser.parse_args(sys.argv)

    dataset = options.dataset
    path = options.path
    load_data = options.load_data
    feature_extract = options.feature_extract

    globalvars.dataset = dataset

    if load_data:
        ds = Dataset(path=path, dataset=dataset)

        print("Writing " + dataset + " data set to file...")
        cPickle.dump(ds, open(dataset + '_db.p', 'wb'))
    else:
        print("Getting data from " + dataset + " data set...")
        ds = cPickle.load(open(dataset + '_db.p', 'rb'))

    if feature_extract:
        functions.feature_extract(ds.data, nb_samples=len(ds.targets), dataset=dataset)

    try:
        trials = Trials()
        best_run, best_model = optim.minimize(model=create_model,
                                              data=get_data,
                                              algo=tpe.suggest,
                                              max_evals=4,
                                              trials=trials)

        U_train, X_train, Y_train, U_test, X_test, Y_test = get_data()

        print(best_run)
        print("Evaluation of best performing model:")
        print(best_model.evaluate([U_test, X_test], Y_test))

        # serialize model to JSON
        best_model_json = best_model.to_json()
        with open('best_model.json', 'w') as json_file:
            json_file.write(best_model_json)
    except IOError:
        print("No training data found, please run with -d [data set], -p [data path], -l and -e "
              "for dumping data at first...")
