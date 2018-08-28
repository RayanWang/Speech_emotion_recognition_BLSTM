import numpy as np


def get_confusion_matrix_one_hot(model_results, truth):
    '''
    model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results, axis=1)
    assert len(predictions) == truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:, actual_class] == 1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class == predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix) == len(truth)
    assert np.sum(confusion_matrix) == np.sum(truth)

    return confusion_matrix
