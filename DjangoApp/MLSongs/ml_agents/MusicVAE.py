from music21 import instrument
from MLSongs.database.db_services import get_songs_by_author
from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.postprocessing_utils import sample, create_midi_with_embedded_durations, \
    change_midi_instrument, midi_to_wav
from MLSongs.ml_agents.preprocessing_utils import get_chords_and_durations_of_instrument, \
     create_mapper_data, create_mapper, encode_notes, clear_encoded_data, \
    parse_everything_together, filter_outliers
from MLSongs.ml_agents.utilities import combine_chords_with_durations, get_key_from_value
import numpy as np


class MusicVAE(MLModelBase):

    def __init__(self, instrument_str):
        if "bass" in instrument_str.lower():
            self.target_instrument_str = "Electric Bass"
            self.target_instrument = instrument.ElectricBass()
            self.instrument_name = "bass"
            super(MusicVAE, self).__init__("MusicVAEBass", "ml_models/VAE_bassdecoder.h5")
        if "guitar" in instrument_str.lower():
            self.target_instrument_str = "Electric Guitar"
            self.target_instrument = instrument.ElectricGuitar()
            self.instrument_name = "guitar"
            super(MusicVAE, self).__init__("MusicVAEGuitar", "ml_models/VAE_guitar_long_decoder.h5")
        self.slice_len = 256
        self.latent_dim = 256

    def preprocess_data(self, data):
        allchords, alldurations = get_chords_and_durations_of_instrument(data, self.target_instrument_str)

        assert (len(allchords) == len(alldurations))

        combined = []
        for i in range(len(allchords)):
            combined.append(combine_chords_with_durations(allchords[i], alldurations[i]))

        self.mapper = create_mapper(create_mapper_data(combined))
        guitar_chords = encode_notes(combined, self.mapper)
        guitar_chords = clear_encoded_data(guitar_chords, self.mapper)

        guitar_input, guitar_output = parse_everything_together(guitar_chords, self.slice_len)

        outlier_constant = 80
        guitar_input, guitar_output, self.mapper_list = filter_outliers(guitar_input, guitar_output, outlier_constant)

        input = np.reshape(np.asarray(guitar_input), (len(guitar_input), self.slice_len, 1))

        input = np.asarray(input) / float(len(self.mapper))

        return input

    def predict(self, input, count, temp):
        songs_in_db_cnt = len(get_songs_by_author(self.db_name))
        to_generate = count

        for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):

            noise = np.random.normal(size=self.latent_dim)
            noise = np.expand_dims(noise, 0)
            pred = self.model.predict(noise)

            predicted = []
            for i in pred:
                for k in i:
                    index = sample(k, temp)
                    if self.mapper_list is not None:  # Idx of the mapper list is the new value, the element is the old value. This is used when I filter for outliers.
                        index = self.mapper_list[index]
                    pred_note = get_key_from_value(index, self.mapper)
                    predicted.append(pred_note)


            midi_path = f'MusicVAE_{self.instrument_name}_{j}.mid'
            create_midi_with_embedded_durations(predicted, target_instrument=self.target_instrument, filename=midi_path)

            change_midi_instrument(midi_path, self.target_instrument)
            midi_to_wav(midi_path, f'static/songs/MusicVAE_{self.instrument_name}_{j}.wav')

            self.save_song_to_db(f'MusicVAE_{self.instrument_name}_{j}.wav')