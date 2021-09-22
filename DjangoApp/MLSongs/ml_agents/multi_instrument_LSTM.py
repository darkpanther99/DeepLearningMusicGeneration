from MLSongs.ml_agents.ml_model_base import MLModelBase
from DjangoApp.secretsconfig import LOCAL_ABSOLUTE_PATH
import numpy as np
from music21 import instrument
from music21.stream import Score
from MLSongs.ml_agents.preprocessing_utils import parse_midi_notes_and_durations, get_chords_and_durations_of_instrument

from MLSongs.ml_agents.postprocessing_utils import generate_multi_instrument_notes, \
    create_midipart_with_durations, midi_to_wav
from MLSongs.ml_agents.preprocessing_utils import create_mapper_data, create_mapper, encode_using_mapper, \
    clear_encoded_data, parse_everything_together, filter_outliers
from MLSongs.ml_agents.preprocessing_utils import encode_notes
from MLSongs.ml_agents.postprocessing_utils import create_drum_part_with_durations

from MLSongs.database.db_services import get_songs_by_author


class MultiInstrumentLSTM(MLModelBase):

    def __init__(self):
        super(MultiInstrumentLSTM, self).__init__('LSTMMultiInstrumentModel', "ml_models/LSTM_multi.h5")
        self.target_instruments_str = ['Electric Guitar', 'Electric Bass', 'Piano']
        self.target_instruments = [instrument.ElectricGuitar(), instrument.ElectricBass(), instrument.Percussion()]
        self.instrument_name = "guitar+bass"
        self.slice_len = 20

    def load_data(self):
        midiparts = parse_midi_notes_and_durations(LOCAL_ABSOLUTE_PATH)
        return midiparts

    def preprocess_data(self, data):
        guitar_chords, guitar_durations = get_chords_and_durations_of_instrument(data, self.target_instruments_str[0])
        bass_chords, bass_durations = get_chords_and_durations_of_instrument(data, self.target_instruments_str[1])
        drum_chords, drum_durations = get_chords_and_durations_of_instrument(data, self.target_instruments_str[2])

        self.guitar_chords = guitar_chords
        self.durations = guitar_durations
        self.bass_chords = bass_chords
        self.drum_chords = drum_chords


        self.guitar_mapper = create_mapper(create_mapper_data(guitar_chords))
        self.guitar_durations_mapper = create_mapper(create_mapper_data(guitar_durations))
        self.bass_mapper = create_mapper(create_mapper_data(bass_chords))
        self.drum_mapper = create_mapper(create_mapper_data(drum_chords))

        guitar_chords = encode_notes(guitar_chords, self.guitar_mapper)
        guitar_durations = encode_notes(guitar_durations, self.guitar_durations_mapper)
        bass_chords = encode_notes(bass_chords, self.bass_mapper)
        drum_chords = encode_notes(drum_chords, self.drum_mapper)

        guitar_chords = clear_encoded_data(guitar_chords, self.guitar_mapper)
        guitar_durations = clear_encoded_data(guitar_durations, self.guitar_durations_mapper)
        bass_chords = clear_encoded_data(bass_chords, self.bass_mapper)
        drum_chords = clear_encoded_data(drum_chords, self.drum_mapper)

        guitar_input, guitar_output = parse_everything_together(guitar_chords, self.slice_len)
        durations_input, durations_output = parse_everything_together(guitar_durations, self.slice_len)
        bass_input, bass_output = parse_everything_together(bass_chords, self.slice_len)
        drum_input, drum_output = parse_everything_together(drum_chords, self.slice_len)

        outlier_constant = 5
        guitar_input, guitar_output, self.guitar_mapper_list = filter_outliers(guitar_input, guitar_output, outlier_constant)
        durations_input, durations_output, self.durations_mapper_list = filter_outliers(durations_input, durations_output, outlier_constant)
        bass_input, bass_output, self.bass_mapper_list = filter_outliers(bass_input, bass_output, 0)
        drum_input, drum_output, self.drum_mapper_list = filter_outliers(drum_input, drum_output, 0)

        guitar_input = np.reshape(np.asarray(guitar_input), (len(guitar_input), self.slice_len, 1))
        durations_input = np.reshape(np.asarray(durations_input), (len(durations_input), self.slice_len, 1))
        bass_input = np.reshape(np.asarray(bass_input), (len(bass_input), self.slice_len, 1))
        drum_input = np.reshape(np.asarray(drum_input), (len(drum_input), self.slice_len, 1))

        return guitar_input, durations_input, bass_input, drum_input

    def predict(self, count, temp):
        #startidx = np.random.randint(0, len(input) - 1)
        #starting_slice = input[startidx]
        #TODO ne az első daltól lehessen csak kezdeni, hanem random daltól
        #Keres egy random dalta mappában, azt beolvassa, első sliceot átalakítja, normálja, stb, és használja
        starting_slice_notes = (np.asarray(encode_using_mapper(self.guitar_chords[0], self.guitar_mapper)) / len(self.guitar_mapper))[:20]
        starting_slice_durations = (np.asarray(encode_using_mapper(self.durations[0], self.guitar_durations_mapper)) / len(
            self.guitar_durations_mapper))[:20]
        starting_slice_bass = (np.asarray(encode_using_mapper(self.bass_chords[0], self.bass_mapper)) / len(self.bass_mapper))[:20]
        starting_slice_drum = (np.asarray(encode_using_mapper(self.drum_chords[0], self.drum_mapper)) / len(self.drum_mapper))[:20]

        songs_in_db_cnt = len(get_songs_by_author(self.db_name))
        to_generate = count

        for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):

            generated_output = generate_multi_instrument_notes(self.model, starting_slice_notes, starting_slice_durations,
                                                               starting_slice_bass, starting_slice_drum, self.guitar_mapper,
                                                               self.guitar_durations_mapper, self.bass_mapper,
                                                               self.drum_mapper, self.guitar_mapper_list, self.durations_mapper_list, temp=temp)

            (guitar_output, bass_output, drum_output) = generated_output


            guitar_part = create_midipart_with_durations(guitar_output, target_instrument=self.target_instruments[0])
            bass_part = create_midipart_with_durations(bass_output, target_instrument=self.target_instruments[1])
            #drum_part = create_drum_part_with_durations(drum_output)

            # TODO dobokat megcsinálni rendesen, hogy dob hangja legyen

            guitar_part.insert(0, self.target_instruments[0])
            bass_part.insert(0, self.target_instruments[1])
            #drum_part.insert(0, self.target_instruments[2])

            full_midi = Score()
            full_midi.insert(0, guitar_part)
            full_midi.insert(0, bass_part)
            #full_midi.insert(0, drum_part)

            midi_path = f'LSTM_{self.instrument_name}_{j}.mid'

            full_midi.write('midi', fp=midi_path)
            midi_to_wav(midi_path, f'static/songs/LSTM_{self.instrument_name}_{j}.wav', True)

            self.save_song_to_db(f'LSTM_{self.instrument_name}_{j}.wav')

