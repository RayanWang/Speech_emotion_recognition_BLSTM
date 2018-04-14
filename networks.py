from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.layers.merge import dot
from keras.models import Model


'''
input_shape: (time_steps, features,)
nb_lstm_cells: 128 memory cells
nb_classes: 7 classes for Berlin data set
'''

attention_layer_name = 'attention_layer'
nb_hidden_units = 512   # number of hidden layer units


def create_softmax_la_network(input_shape, nb_lstm_cells=128, nb_classes=7):
    # Logistic regression for learning the attention parameters with a standalone feature as input
    input_attention = Input(shape=(nb_lstm_cells * 2,))
    u = Dense(nb_lstm_cells * 2, activation='softmax', name=attention_layer_name)(input_attention)

    # Bi-directional Long Short-Term Memory for learning the temporal aggregation
    input_feature = Input(shape=input_shape)
    x = Masking(mask_value=-100.0)(input_feature)
    x = Dense(nb_hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x)

    # To compute the final weights for the frames which sum to unity
    alpha = dot([u, y], axes=-1)  # inner prod.
    alpha = Activation('softmax')(alpha)

    # Weighted pooling to get the utterance-level representation
    z = dot([alpha, y], axes=1)

    # Get posterior probability for each emotional class
    output = Dense(nb_classes, activation='softmax')(z)

    return Model(inputs=[input_attention, input_feature], outputs=output)
