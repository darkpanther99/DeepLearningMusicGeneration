from MLSongs.ml_agents.ml_model_base import MLModelBase
from MLSongs.ml_agents.postprocessing_utils import create_midi_with_embedded_durations, change_midi_instrument, midi_to_wav
from MLSongs.database.models import Song, MLModel
import os
from music21 import instrument


class Music_GPT_2(MLModelBase):
    def __init__(self):
        self.target_instrument = instrument.ElectricGuitar()

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
        #Ha a db-ben annyi gpt zene van, mint ahány a gpt-outputs folderben, akkor ne generáljon újat
        for idx, note_sequence in enumerate(input):
            midi_path = f'GPT-2_OUTPUT{idx}.mid'
            try:
                #Exceptions can occur, because the GPT-2 model makes mistakes while generating text, resulting in invalid MIDI notes.
                create_midi_with_embedded_durations(note_sequence, filename=midi_path)

                change_midi_instrument(midi_path, self.target_instrument)
                midi_to_wav(midi_path, f'static/songs/GPT-2_OUTPUT{idx}.wav', True)

                self.save_song_to_db(f'GPT-2_OUTPUT{idx}.wav')

            except:
                #If an exception is the case I just ignore that sample
                pass

    def save_song_to_db(self, song_path):
        # Since I only have 1 LSTMModel record, that will be the MLModel.
        mc_author = MLModel.objects.filter(name="GPT-2Model").first()

        if not mc_author:
            mc_author = MLModel(name="GPT-2Model")
            mc_author.save()

        # The song's title is the path without the wav extension.
        s = Song(title=song_path[:-4], author=mc_author, path=song_path)
        s.save()