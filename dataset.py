import os
import itertools
import librosa


class Dataset:

    def __init__(self, path, name_database):
        self.database = name_database
        if name_database == "berlin":
            self.classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
            self.get_berlin_dataset(path)

    def get_berlin_dataset(self, path):
        males = ['03', '10', '11', '12', '15']
        females = ['08', '09', '13', '14', '16']
        classes = {v: k for k, v in self.classes.iteritems()}
        self.targets = []
        self.data = []
        self.train_sets = []
        self.test_sets = []
        get_data = True
        for speak_test in itertools.product(males, females):  # test_couples:
            i = 0
            train = []
            test = []
            for audio in os.listdir(path):
                audio_path = os.path.join(path, audio)
                y, sr = librosa.load(audio_path, sr=16000)
                if get_data:
                    self.data.append((y, sr))
                    self.targets.append(classes[audio[5]])
                if audio[:2] in speak_test:
                    test.append(i)
                else:
                    train.append(i)
                i = i + 1
            self.train_sets.append(train)
            self.test_sets.append(test)
            get_data = False
