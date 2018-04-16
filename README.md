# Speech_emotion_recognition_BLSTM

Bidirectional LSTM network for speech emotion recognition.

Environment: Python 2.7

# Dependencies

- [Tensorflow(1.6)](https://github.com/tensorflow/tensorflow/tree/r1.6) for the backend of keras
- [keras(2.1.5)](https://github.com/keras-team/keras) for building/training the Bidirectional LSTM network
- [librosa](https://github.com/librosa/librosa) for audio resampling
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) for feature engineering
- [scikit learn](https://github.com/scikit-learn/scikit-learn) for k-fold cross validation

# Datasets

- [Berlin speech dataset](http://emodb.bilderbar.info/download/)

# How to use

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset path
--cross_validation | -c | do cross validation
--load_data | -l | load dataset and dump the data stream to a .p file
--feature_extract | -e | extract features from data and dump to a .p file
--speaker_independence | -s | cross validation is made using different actors for train and test sets

Example:

    python model_training.py -d 'berlin' -p [berlin db path] -l -e 
   
- The first time you run the script, -l and -e options are mandatory since you need to load data and extract features. Every time you change the training data and/or the method of feature engineering, you have to specify -l and/or -e respectively to update your .p files.

- You can use -c for k-fold cross validation with -s if you need speaker independent recognition.

# Experimental result

- The average accuracy is about 68.60%(+/- 1.88%) with the latest code.

# References

- Seyedmahdad Mirsamadi,Emad Barsoum and Cha Zhang, "AUTOMATIC SPEECH EMOTION RECOGNITION USING RECURRENT NEURAL
NETWORKS WITH LOCAL ATTENTION", IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), 2017

- [Video](https://www.youtube.com/watch?v=NItzgTQ9lvw) from Microsoft Research
