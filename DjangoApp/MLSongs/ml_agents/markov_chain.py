from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.utilities import get_key_from_value
from MLSongs.ml_agents.postprocessing_utils import create_midi_with_durations, midi_to_wav, change_midi_instrument
from MLSongs.ml_agents.preprocessing_utils import parse_midi_notes_and_durations, create_mapper, encode_using_mapper, get_chords_and_durations_of_instrument
import numpy as np
import mchmm as mc
import random
from music21 import instrument
import pickle
from MLSongs.database.db_services import get_songs_by_author


class MarkovModel(MLModelBase):

    def __init__(self, instrument_str):
        if 'bass' in instrument_str.lower():
            self.target_instrument_str = 'Electric Bass'
            self.target_instrument = instrument.ElectricBass()
            self.instrument_name = 'bass'
            db_name = "MarkovBass"
        if 'guitar' in instrument_str.lower():
            self.target_instrument_str = 'Electric Guitar'
            self.target_instrument = instrument.ElectricGuitar()
            self.instrument_name = 'guitar'
            db_name = "MarkovGuitar"

        super(MarkovModel, self).__init__(db_name, "")


    def preprocess_data(self, data):

        allchords, alldurations = get_chords_and_durations_of_instrument(data, self.target_instrument_str)

        chord_mapper_data = []
        for i in allchords:
            for j in i:
                chord_mapper_data.append(j)
        self.mapper = create_mapper(chord_mapper_data)

        duration_mapper_data = []
        for i in alldurations:
            for j in i:
                duration_mapper_data.append(j)
        self.duration_mapper = create_mapper(duration_mapper_data)

        encoded_chords = []
        durationsdata = []

        for c in allchords:
            encoded = encode_using_mapper(c, self.mapper)
            for i in encoded:
                encoded_chords.append(i)

        for d in alldurations:
            encoded = encode_using_mapper(d, self.duration_mapper)
            for i in encoded:
                durationsdata.append(i)

        encoded_chord_string = []
        encoded_duration_string = []
        for i in encoded_chords:
            encoded_chord_string.append(str(i))

        for i in durationsdata:
            encoded_duration_string.append(str(i))

        return encoded_chord_string, encoded_duration_string

    def build_model(self, encoded_chord_string, encoded_duration_string):
        with open(f'ml_models/markov_{self.instrument_name}_chords', 'rb') as outp:
            self.chordsmc = pickle.load(outp)
        with open(f'ml_models/markov_{self.instrument_name}_durations', 'rb') as outp:
            self.durationmc = pickle.load(outp)

    def build_and_save_model(self, encoded_chord_string, encoded_duration_string):
        self.chordsmc = mc.MarkovChain().from_data(encoded_chord_string)
        with open('ml_models/markov_bass_chords', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.chordsmc, outp, pickle.HIGHEST_PROTOCOL)
        self.durationmc = mc.MarkovChain().from_data(encoded_duration_string)
        with open('ml_models/markov_bass_durations', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.durationmc, outp, pickle.HIGHEST_PROTOCOL)

    def generate_music(self, encoded_chord_string, encoded_duration_string, count, length=200):
        songs_in_db_cnt = len(get_songs_by_author(self.db_name))
        to_generate = count

        for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):
            chord_states = None
            duration_states = None
            # This is an ugly hack, don't try anything like this at home.
            # chordsmc.simulate() throws an exception 90% of the time, because of floating point rounding errors, because my transition matrix is too huge.
            # A workaround is this ugly while True loop, which will spin the CPU until we get an errorless simulation.
            while chord_states is None:
                try:
                    ids, chord_states = self.chordsmc.simulate(length,tf=np.asarray(self.chordsmc.observed_matrix).astype('float64'),start=random.choice(encoded_chord_string))
                except:
                    pass

            while duration_states is None:
                try:
                    durids, duration_states = self.durationmc.simulate(length, tf=np.asarray(self.durationmc.observed_matrix).astype('float64'), start=random.choice(encoded_duration_string))
                except:
                    pass

            music = []
            musicdurations = []

            for i in chord_states:
                note = get_key_from_value(int(i), self.mapper)
                music.append(note)

            for i in duration_states:
                duration = get_key_from_value(int(i), self.duration_mapper)
                musicdurations.append(duration)

            midi_path = f'Markov_{self.instrument_name}_{j}.mid'
            create_midi_with_durations(music, musicdurations, self.target_instrument, midi_path)
            change_midi_instrument(midi_path, self.target_instrument)
            midi_to_wav(midi_path, f'static/songs/Markov_{self.instrument_name}_{j}.wav')

            self.save_song_to_db(f'Markov_{self.instrument_name}_{j}.wav')


if __name__ == '__main__':
    MarkovModel()