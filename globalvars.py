# This file is used for find_best_model.py

globalVar = 0

dataset = 'berlin'

max_len = 1024
nb_features = 34

nb_attention_param = 256
attention_init_value = 1.0 / 256
nb_hidden_units = 512
dropout_rate = 0.5
nb_lstm_cells = 128
nb_classes = 7

frame_size = 0.025  # 25 msec segments
step = 0.01     # 10 msec time step
