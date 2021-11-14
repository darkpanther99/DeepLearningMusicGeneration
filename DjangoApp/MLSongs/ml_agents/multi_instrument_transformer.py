import random
from MLSongs.ml_agents.ml_model_base import MLModelBase
import numpy as np
from keras.models import load_model
from music21 import instrument, note
from music21.stream import Score
from MLSongs.ml_agents.preprocessing_utils import parse_midi_notes_and_durations, get_chords_and_durations_of_instrument

from MLSongs.ml_agents.postprocessing_utils import generate_multi_instrument_notes, \
    create_midipart_with_durations, midi_to_wav
from MLSongs.ml_agents.preprocessing_utils import create_mapper_data, create_mapper, encode_using_mapper, \
    clear_encoded_data, parse_everything_together, filter_outliers
from MLSongs.ml_agents.preprocessing_utils import encode_notes
from MLSongs.ml_agents.postprocessing_utils import create_drum_part_with_durations
from MLSongs.database.db_services import get_songs_by_author
from MLSongs.ml_agents.utilities import combine_chords_with_durations
from MLSongs.ml_agents.postprocessing_utils import generate_notes
from MLSongs.database.db_services import get_model_with_insert
from MLSongs.ml_agents.Attention import DynamicPositionEmbedding


class MultiInstrumentTransformer(MLModelBase):

    def __init__(self):
        super(MultiInstrumentTransformer, self).__init__('TransformerMultiInstrumentModel', "ml_models/Transformer_guitar.h5;ml_models/Transformer_bass_short.h5")
        self.target_instruments_str = ['Electric Guitar', 'Electric Bass']
        self.target_instruments = [instrument.ElectricGuitar(), instrument.ElectricBass()]
        self.instrument_name = "guitar+bass"
        self.slice_len = 20

    def preprocess_data(self, data):
        guitar_chords_raw, guitar_durations_raw = get_chords_and_durations_of_instrument(data, self.target_instruments_str[0])
        bass_chords_raw, bass_durations = get_chords_and_durations_of_instrument(data, self.target_instruments_str[1])

        combined_guitar = []
        for i in range(len(guitar_chords_raw)):
            combined_guitar.append(combine_chords_with_durations(guitar_chords_raw[i], guitar_durations_raw[i]))
        self.guitar_mapper = create_mapper(create_mapper_data(combined_guitar))

        combined_bass = []
        for i in range(len(bass_chords_raw)):
            combined_bass.append(combine_chords_with_durations(bass_chords_raw[i], bass_durations[i]))
        self.bass_mapper = create_mapper(create_mapper_data(combined_bass))

        guitar_chords = encode_notes(combined_guitar, self.guitar_mapper)
        bass_chords = encode_notes(combined_bass, self.bass_mapper)

        guitar_chords = clear_encoded_data(guitar_chords, self.guitar_mapper)
        bass_chords = clear_encoded_data(bass_chords, self.bass_mapper)

        guitar_input, guitar_output = parse_everything_together(guitar_chords, self.slice_len)
        bass_input, bass_output = parse_everything_together(bass_chords, self.slice_len)

        outlier_constant = 40
        guitar_input, guitar_output, self.guitar_mapper_list = filter_outliers(guitar_input, guitar_output, outlier_constant)
        bass_input, bass_output, self.bass_mapper_list = filter_outliers(bass_input, bass_output, outlier_constant)

        guitar_input = np.reshape(np.asarray(guitar_input), (len(guitar_input), self.slice_len, 1))
        bass_input = np.reshape(np.asarray(bass_input), (len(bass_input), self.slice_len, 1))

        return guitar_chords_raw, bass_chords_raw

    def build_model(self):
        ml_author = get_model_with_insert(self.db_name, self.path)
        split_path = ml_author.path.split(';')
        self.guitar_model = load_model(split_path[0], custom_objects={'DynamicPositionEmbedding': DynamicPositionEmbedding})
        self.bass_model = load_model(split_path[1], custom_objects={'DynamicPositionEmbedding': DynamicPositionEmbedding})

    def predict(self, data, count, temp, length=500):

        songs = list(set([i.song for i in data]))

        bug = True
        while bug:
            try:
                condition = True
                while condition:
                    try:
                        random_song = random.choice(songs)
                        slice_by_instrument = dict(zip(self.target_instruments_str, [[] for i in self.target_instruments_str]))
                        for j in self.target_instruments_str:
                            for i in data:
                                if i.song == random_song and i.instrument == j:
                                    slice_by_instrument[j].append(i)

                        slice_by_instrument_without_rests = dict(zip(self.target_instruments_str, [[] for i in self.target_instruments_str]))

                        for i in slice_by_instrument.keys():
                            for song in slice_by_instrument[i]:
                                if not isinstance(song.chords[0], note.Rest):
                                    slice_by_instrument_without_rests[i].append(song)
                            if len(slice_by_instrument_without_rests[i]) != 0:
                                slice_by_instrument[i] = random.choice(slice_by_instrument_without_rests[i])
                            else:
                                slice_by_instrument[i] = random.choice(slice_by_instrument[i])

                        condition = False
                    except IndexError:
                        continue

                guitar_chords = slice_by_instrument['Electric Guitar'].chords
                guitar_durations = slice_by_instrument['Electric Guitar'].durations
                bass_chords = slice_by_instrument['Electric Bass'].chords
                bass_durations = slice_by_instrument['Electric Bass'].durations

                combined_guitar = combine_chords_with_durations(guitar_chords, guitar_durations)
                combined_bass = combine_chords_with_durations(bass_chords, bass_durations)

                starting_slice_notes = (np.asarray(encode_using_mapper(combined_guitar, self.guitar_mapper)))[:20]
                starting_slice_bass = (np.asarray(encode_using_mapper(combined_bass, self.bass_mapper)))[:20]

                songs_in_db_cnt = len(get_songs_by_author(self.db_name))
                to_generate = count

                for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):

                    generated_guitar = generate_notes(self.guitar_model, starting_slice_notes, self.guitar_mapper, mapperlist = self.guitar_mapper_list, temp=temp, length = length, normalize = False, random_start = False)
                    generated_bass = generate_notes(self.bass_model, starting_slice_bass, self.bass_mapper, mapperlist = self.bass_mapper_list, temp=temp, length = length, normalize = False, random_start = False)

                    guitar_part = create_midipart_with_durations(generated_guitar, target_instrument=self.target_instruments[0])
                    bass_part = create_midipart_with_durations(generated_bass, target_instrument=self.target_instruments[1])

                    guitar_part.insert(0, self.target_instruments[0])
                    bass_part.insert(0, self.target_instruments[1])

                    full_midi = Score()
                    full_midi.insert(0, guitar_part)
                    full_midi.insert(0, bass_part)

                    midi_path = f'Transformer_{self.instrument_name}_{j}.mid'

                    full_midi.write('midi', fp=midi_path)
                    midi_to_wav(midi_path, f'static/songs/Transformer_{self.instrument_name}_{j}.wav')

                    self.save_song_to_db(f'Transformer_{self.instrument_name}_{j}.wav')
                bug = False
            except:
                continue