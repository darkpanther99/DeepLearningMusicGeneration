from DjangoApp.secretsconfig import LOCAL_ABSOLUTE_PATH
from MLSongs.database.models import Song, MLModel
from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.utilities import parse_midi_notes_and_durations,\
    get_chords_and_durations_of_instrument, create_mapper, encode_using_mapper, get_key_from_value,\
    create_midi_with_durations, midi_to_wav, change_midi_instrument, combine_chords_with_durations, most_frequent,\
    parse_everything_together
from collections import Counter
import numpy as np
import mchmm as mc
import random
from music21 import instrument
import pickle
from tensorflow.keras.models import load_model

class LSTMModel(MLModelBase):

    def __init__(self):
        self.target_instrument_str = 'Electric Guitar'
        self.target_instrument = instrument.ElectricGuitar()
        self.slice_len = 10
        super(LSTMModel, self).__init__()
        data = self.load_data()
        input = self.preprocess_data(data)

    def load_data(self):
        midiparts = parse_midi_notes_and_durations(LOCAL_ABSOLUTE_PATH)
        return midiparts

    def preprocess_data(self, data):
        allchords, alldurations = get_chords_and_durations_of_instrument(data, self.target_instrument_str)

        assert (len(allchords) == len(alldurations))

        combined = []
        for i in range(len(allchords)):
            combined.append(combine_chords_with_durations(allchords[i], alldurations[i]))

        mapperdata = []

        for i in combined:
            for j in i:
                mapperdata.append(j)

        self.mapper = create_mapper(mapperdata)

        encoded_data = []

        for c in combined:
            encoded = encode_using_mapper(c, self.mapper)
            encoded_data.append(encoded)

        restkeysvalues = []
        for j in self.mapper.keys():
            if ('rest' in j):
                restkeysvalues.append(self.mapper[j])

        cleared_encoded_data = []

        for i in range(len(encoded_data)):
            if most_frequent(encoded_data[i]) not in restkeysvalues:
                cleared_encoded_data.append(encoded_data[i])

        input, output = parse_everything_together(cleared_encoded_data, self.slice_len)

        outputcnt = Counter(output)

        outliers = []
        OUTLIER_CONSTANT = 40

        for i in outputcnt.keys():
            if outputcnt[i] < OUTLIER_CONSTANT:
                outliers.append(i)

        assert (len(input) == len(output))

        newinput = []
        newoutput = []

        for i in range(len(output)):
            if (output[i] not in outliers):
                newinput.append(input[i])
                newoutput.append(output[i])

        input = newinput
        output = newoutput

        assert (len(input) == len(output))

        self.mapper_list = []  # Idx of the mapper list is the new value, the element is the old value.
        new_output_elements = set(output)

        for i in new_output_elements:
            self.mapper_list.append(i)

        newoutput = []

        for i in output:
            newoutput.append(self.mapper_list.index(i))

        output = newoutput

        input = np.reshape(np.asarray(input), (len(input), self.slice_len, 1))

        input = np.asarray(input) / float(len(self.mapper))

        return input

    def build_model(self):
        pass

    def predict(self, input):
        startidx = np.random.randint(0, len(input) - 1)
        starting_slice = input[startidx]