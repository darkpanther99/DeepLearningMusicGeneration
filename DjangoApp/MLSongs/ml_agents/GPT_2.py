from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.postprocessing_utils import create_midi_with_embedded_durations, change_midi_instrument, midi_to_wav
from MLSongs.database.models import Song, MLModel
import os
from music21 import instrument


class Music_GPT_2(MLModelBase):
    def __init__(self):
        super(Music_GPT_2, self).__init__("GPT-2Model", "")
        self.target_instrument = instrument.ElectricGuitar()
        self.instrument_name = "guitar"

    def load_data(self):
        text_path = 'gpt_outputs'
        all_outputs = []
        for file in os.listdir(text_path):
            f = open(os.path.join(text_path, file), "rt")
            gpt_output = f.read()[27:]
            f.close()
            all_outputs.append(gpt_output)


        return  all_outputs

    def preprocess_data(self, data):
        all_notes = []
        for gpt_output in data:
            notes = gpt_output.split(" ")[1:-2]
            all_notes.append(notes)

        return all_notes

    def predict(self, input, count):
        for idx, note_sequence in enumerate(input):
            midi_path = f'GPT-2_{self.instrument_name}_{idx}.mid'
            try:
                #Exceptions can occur, because the GPT-2 model makes mistakes while generating text, resulting in invalid MIDI notes.
                create_midi_with_embedded_durations(note_sequence, filename=midi_path)

                change_midi_instrument(midi_path, self.target_instrument)
                midi_to_wav(midi_path, f'static/songs/GPT-2_{self.instrument_name}_{idx}.wav', False)

                self.save_song_to_db(f'GPT-2_{self.instrument_name}_{idx}.wav')

            except:
                #If an exception is the case I just ignore that sample
                pass