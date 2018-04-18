# Speech_emotion_recognition_BLSTM

Bidirectional LSTM network for speech emotion recognition.

Environment:

- Python 2.7
- NVIDIA GTX 1060 6GB
- conda version 4.5

# Dependencies

- [Tensorflow(1.6)](https://github.com/tensorflow/tensorflow/tree/r1.6) for the backend of keras
- [keras(2.1.5)](https://github.com/keras-team/keras) for building/training the Bidirectional LSTM network
- [librosa](https://github.com/librosa/librosa) for audio resampling
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) for feature engineering
- [scikit learn](https://github.com/scikit-learn/scikit-learn) for k-fold cross validation
- [Hyperas](https://github.com/maxpumperla/hyperas) for fine-tuning hyper parameters and find best model

# Datasets

- [Berlin speech dataset](http://emodb.bilderbar.info/download/)

# Usage

- You have to prepare at least two different sets of data, one for find the best model and the other for cross validation.

1. find_best_model:

    python find_best_model.py -d "berlin" -p [berlin data path] -l -e

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset path
--load_data | -l | load dataset and dump the data stream to a .p file
--feature_extract | -e | extract features from data and dump to a .p file

- The first time you run the script, -l and -e options are mandatory since you need to load data and extract features.
- Every time you change the training data and/or the method of feature engineering, you have to specify -l and/or -e respectively to update your .p files.
- You can also modify the code for tuning other hyper parameters.

2. model_testing:

    python model_testing.py -d "dafex" -p [dafex data path] -w [weights path] -l -e

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset path
--weights_path | -w | weights path
--load_data | -l | load dataset and dump the data stream to a .p file
--feature_extract | -e | extract features from data and dump to a .p file

- Please be careful not to use the data set the same with the best model you tuned before.
- Parameters of -d, -p, -l, -e are all the same in 1.

3. model_cross_validation:

    python model_cross_validation.py -d "berlin" -p [berlin data path] -l -e

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset path
--load_data | -l | load dataset and dump the data stream to a .p file
--feature_extract | -e | extract features from data and dump to a .p file
--speaker_indipendence | -s | cross validation is made using different actors for train and test sets

- -s for cross validation of different actors.
- Parameters of -d, -p, -l, -e are all the same in 1.

# Experimental result

- Use hyperas for tuning optimizer, batch_size and epochs, the remaining parameters are the values applied to this paper.
- The average accuracy is about 68.60%(+/- 1.88%, through 10-fold cross validation).

# References

- Seyedmahdad Mirsamadi,Emad Barsoum and Cha Zhang, "AUTOMATIC SPEECH EMOTION RECOGNITION USING RECURRENT NEURAL
NETWORKS WITH LOCAL ATTENTION", IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), 2017

- [Video](https://www.youtube.com/watch?v=NItzgTQ9lvw) from Microsoft Research
