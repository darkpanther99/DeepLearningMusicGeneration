from DjangoApp.secretsconfig import LOCAL_ABSOLUTE_PATH
from MLSongs.database.models import Song, MLModel
from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.utilities import get_key_from_value,\
    combine_chords_with_durations, most_frequent
from MLSongs.ml_agents.postprocessing_utils import generate_notes, create_midi_with_embedded_durations, create_midi_with_durations, midi_to_wav, change_midi_instrument
from MLSongs.ml_agents.preprocessing_utils import parse_midi_notes_and_durations, create_mapper, encode_using_mapper, get_chords_and_durations_of_instrument, parse_everything_together
from collections import Counter
import numpy as np
from music21 import instrument
from tensorflow.keras.models import load_model

class LSTMModel(MLModelBase):

    def __init__(self):
        super(LSTMModel, self).__init__()
        self.target_instrument_str = 'Electric Guitar'
        self.target_instrument = instrument.ElectricGuitar()
        self.slice_len = 10


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
        ml_author = MLModel.objects.filter(name=self.db_name).first()

        if not ml_author:
            ml_author = MLModel(name=self.db_name, path=self.path)
            ml_author.save()

        self.model = load_model(ml_author.path)

    def predict(self, input, count, temp):
        #startidx = np.random.randint(0, len(input) - 1)
        #starting_slice = input[startidx]
        songs_in_db_cnt = len(Song.objects.all())
        to_generate = count

        for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):

            generated_output = generate_notes(self.model, input, self.mapper, self.mapper_list, temp=temp)

            midi_path = f'LSTM_OUTPUT{j}.mid'
            create_midi_with_embedded_durations(generated_output, target_instrument=instrument.ElectricGuitar(), filename=midi_path)

            change_midi_instrument(midi_path, self.target_instrument)
            midi_to_wav(midi_path, f'static/songs/LSTM_OUTPUT{j}.wav', True)

            self.save_song_to_db(f'LSTM_OUTPUT{j}.wav')

    def save_song_to_db(self, song_path):
        # Since I only have 1 LSTMModel record, that will be the MLModel.
        mc_author = MLModel.objects.filter(name="LSTMModel").first()

        if not mc_author:
            mc_author = MLModel(name="LSTMModel", path = "ml_models/guitarLSTM1.h5")
            mc_author.save()

        # The song's title is the path without the wav extension.
        s = Song(title=song_path[:-4], author=mc_author, path=song_path)
        s.save()