# Speech_emotion_recognition_BLSTM

Bidirectional LSTM network for speech emotion recognition.

Environment:

- Python 2.7/3.6
- NVIDIA Geforce GTX 1060 6GB
- Conda version 4.5

# Dependencies

- [Tensorflow(1.6)](https://github.com/tensorflow/tensorflow/tree/r1.6) for the backend of keras
- [keras(2.1.5)](https://github.com/keras-team/keras) for building/training the Bidirectional LSTM network
- [librosa](https://github.com/librosa/librosa) for audio resampling
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) for feature engineering
- [scikit learn](https://github.com/scikit-learn/scikit-learn) for k-fold cross validation
- [Hyperas](https://github.com/maxpumperla/hyperas) for fine-tuning hyper parameters and find best model
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) for sentence extraction
- [pydub](https://github.com/jiaaro/pydub) for wav extraction

# Datasets

- [Berlin speech dataset](http://emodb.bilderbar.info/download/)

# Usage

- Since the function "stFeatureSpeed" in pyAudioAnalysis is default unworkable, you have to modify the code in audioFeatureExtraction.py (for index related issue, just cast the value type to integer; for the issue in method stHarmonic, cast M to integer(M = int(M)).
- If you run the code in python 3, please upgrade pyAudioAnalysis to the latest version that compatible with python 3.
- You have to prepare at least two different sets of data, one for find the best model and the other for testing.

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset or the predicted data path
--load_data | -l | load dataset and dump the data stream to a .p file
--feature_extract | -e | extract features from data and dump to a .p file
--model_path | -m | the model path you want to load
--nb_classes | -c | the number of classes of your data
--speaker_indipendence | -s | cross validation is made using different actors for train and test sets

Example find_best_model.py:

    python find_best_model.py -d "berlin" -p [berlin data path] -l -e -c 7

- The first time you run the script, -l and -e options are mandatory since you need to load data and extract features.
- Every time you change the training data and/or the method of feature engineering, you have to specify -l and/or -e respectively to update your .p files.
- You can also modify the code for tuning other hyper parameters.

Example prediction.py:

    python prediction.py -p [data path] -m [model path] -l -e -c 7

Example model_cross_validation.py:

    python model_cross_validation.py -d "berlin" -p [berlin data path] -l -e -c 7

- Use -s for k-fold cross validation in different actors.

# Experimental result

- Use hyperas for tuning optimizers, batch_size and epochs, the remaining parameters are the values applied to the paper below.
- The average accuracy is about 68.60%(+/- 1.88%, through 10-fold cross validation, using Berlin dataset).

# References

- S. Mirsamadi, E. Barsoum, and C. Zhang, “Automatic speech emotion recognition using recurrent neural networks with local attention,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, U.S.A., Mar. 2017, IEEE, pp. 2227–2231.

- Fei Tao, Gang Liu, “Advanced LSTM: A Study about Better Time Dependency Modeling in Emotion Recognition,” Submitted to 2018 IEEE International Conference on Acoustics, Speech and Signal Processing.

- [Video](https://www.microsoft.com/en-us/research/video/automatic-speech-emotion-recognition-using-recurrent-neural-networks-local-attention/) from Microsoft Research

# Future work

- The training data I list above (Berlin) may insufficient, the validation accuracy and loss can't be improved while the training result is not good.
- Given sufficient training examples, the parameters of short-term characterization, long-term aggregation, and the attention model can be jointly optimized for best performance.
- Update the current network architecture to improve the accuracy (already in progress).
