from MLSongs.ml_agents.Attention import DynamicPositionEmbedding, AttentionModel
from MLSongs.ml_agents.postprocessing_utils import generate_notes, create_midi_with_embedded_durations, create_midi_with_durations, midi_to_wav, change_midi_instrument
from music21 import instrument
from MLSongs.database.db_services import get_songs_by_author, get_model_with_insert


class TransformerModel(AttentionModel):

    def __init__(self, instrument_str):
        if "guitar" in instrument_str:
            self.target_instrument_str = 'Electric Guitar'
            self.target_instrument = instrument.ElectricGuitar()
            self.instrument_name = "guitar"
            super(AttentionModel, self).__init__("TransformerModel", "ml_models/Transformer_guitar.h5")
            self.slice_len = 20
        elif "bass" in instrument_str:
            self.target_instrument_str = 'Electric Bass'
            self.target_instrument = instrument.ElectricBass()
            self.instrument_name = "bass"
            super(AttentionModel, self).__init__("TransformerModelBass", "ml_models/Transformer_bass_short.h5")
            self.slice_len = 20

    def predict(self, input, count, temp, length=500):
        songs_in_db_cnt = len(get_songs_by_author(self.db_name))
        to_generate = count

        for j in range(songs_in_db_cnt, songs_in_db_cnt + to_generate):

            generated_output = generate_notes(self.model, input, self.mapper, self.mapper_list, temp=temp, length = length, normalize = False)

            midi_path = f'Transformer_{self.instrument_name}_{j}.mid'
            create_midi_with_embedded_durations(generated_output, target_instrument=self.target_instrument, filename=midi_path)

            change_midi_instrument(midi_path, self.target_instrument)
            midi_to_wav(midi_path, f'static/songs/Transformer_{self.instrument_name}_{j}.wav', False)

            self.save_song_to_db(f'Transformer_{self.instrument_name}_{j}.wav')