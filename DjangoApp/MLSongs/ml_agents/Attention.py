from DjangoApp.secretsconfig import LOCAL_ABSOLUTE_PATH
from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.utilities import get_key_from_value,\
    combine_chords_with_durations, most_frequent
from MLSongs.ml_agents.postprocessing_utils import generate_notes, create_midi_with_embedded_durations, create_midi_with_durations, midi_to_wav, change_midi_instrument
from MLSongs.ml_agents.preprocessing_utils import parse_midi_notes_and_durations, create_mapper, encode_using_mapper, get_chords_and_durations_of_instrument,\
    parse_everything_together, create_mapper_data, encode_notes, clear_encoded_data, filter_outliers
import numpy as np
from music21 import instrument
from MLSongs.database.db_services import get_songs_by_author, get_model_with_insert
from keras.models import load_model

from tensorflow.keras.layers import Layer
import tensorflow as tf
import math as m

#Custom layer, which is needed to load the model
class DynamicPositionEmbedding(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embedding_dim=256
        max_seq=2048
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding[:,:inputs.shape[1],:])


class AttentionModel(MLModelBase):

    def __init__(self, instrument_str):
        if "guitar" in instrument_str:
            self.target_instrument_str = 'Electric Guitar'
            self.target_instrument = instrument.ElectricGuitar()
            self.instrument_name = "guitar"
            super(AttentionModel, self).__init__("AttentionModel", "ml_models/Attention_guitar.h5")
            self.slice_len = 20
        elif "bass" in instrument_str:
            self.target_instrument_str = 'Electric Bass'
            self.target_instrument = instrument.ElectricBass()
            self.instrument_name = "bass"
            super(AttentionModel, self).__init__("AttentionModelBass", "ml_models/Attention_bass.h5")
            self.slice_len = 20

    def build_model(self):
        ml_author = get_model_with_insert(self.db_name, self.path)
        self.model = load_model(ml_author.path, custom_objects={'DynamicPositionEmbedding': DynamicPositionEmbedding})


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

        outlier_constant = 40
        guitar_input, guitar_output, self.mapper_list = filter_outliers(guitar_input, guitar_output, outlier_constant)

        input = np.reshape(np.asarray(guitar_input), (len(guitar_input), self.slice_len, 1))

        return input


    def predict(self, input, count, temp):
        #startidx = np.random.randint(0, len(input) - 1)
        #starting_slice = input[startidx]
        songs_in_db_cnt = len(get_songs_by_author(self.db_name))
        to_generate = count

        for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):

            generated_output = generate_notes(self.model, input, self.mapper, self.mapper_list, temp=temp)

            midi_path = f'Attention_{self.instrument_name}_{j}.mid'
            create_midi_with_embedded_durations(generated_output, target_instrument=self.target_instrument, filename=midi_path)

            change_midi_instrument(midi_path, self.target_instrument)
            midi_to_wav(midi_path, f'static/songs/Attention_{self.instrument_name}_{j}.wav', True)

            self.save_song_to_db(f'Attention_{self.instrument_name}_{j}.wav')
