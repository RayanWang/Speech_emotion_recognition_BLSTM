from optparse import OptionParser
from keras.utils import to_categorical
from keras.models import model_from_json
from functions import Dataset, feature_extraction, globalvars

import numpy as np
import sys
import cPickle


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--testset', dest='testset', default='dafex')
    parser.add_option('-p', '--testset_path', dest='path', default='')
    parser.add_option('-l', '--load_data', action='store_true', dest='load_data')
    parser.add_option('-e', '--feature_extract', action='store_true', dest='feature_extract')
    parser.add_option('-w', '--weights_path', dest='weights_path', default='')

    (options, args) = parser.parse_args(sys.argv)

    testset = options.testset
    path = options.path
    load_data = options.load_data
    feature_extract = options.feature_extract
    weights_path = options.weights_path

    if load_data:
        print("Loading data from " + testset + " data set...")
        if testset not in ('dafex', 'berlin'):
            sys.exit("Dataset not registered. Please create a method to read it")

        db = Dataset(path, testset, decode=False)

        print("Dumping " + testset + " data set to file...")
        cPickle.dump(db, open(testset + '_db.p', 'wb'))
    else:
        print("Getting data from " + testset + " data set...")
        db = cPickle.load(open(testset + '_db.p', 'rb'))

    nb_samples = len(db.targets)
    print("Number of samples: " + str(nb_samples))

    if feature_extract:
        f_global = feature_extraction.training_extract(db.data, nb_samples=nb_samples, dataset=testset)
    else:
        print("Getting features from files...")
        f_global = cPickle.load(open(testset + '_features.p', 'rb'))

    y = np.array(db.targets)
    y = to_categorical(y, num_classes=globalvars.nb_classes)

    u = np.full((f_global.shape[0], globalvars.nb_attention_param), globalvars.attention_init_value,
                dtype=np.float64)

    # load json and create model
    with open('best_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    best_model = model_from_json(loaded_model_json)

    # load weights into new model
    best_model.load_weights(weights_path)

    # evaluate the model
    scores = best_model.evaluate([u, f_global], y, verbose=1)

    print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1] * 100))
