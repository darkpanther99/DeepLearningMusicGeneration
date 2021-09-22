from keras.models import load_model
from MLSongs.database.db_services import get_model_with_insert, save_song

class MLModelBase:


    def __init__(self, *args):
        if len(args) == 2:
            self.db_name = args[0]
            self.path = args[1]

    def load_data(self):
        pass

    def preprocess_data(self, data):
        pass

    def build_model(self):
        ml_author = get_model_with_insert(self.db_name, self.path)
        self.model = load_model(ml_author.path)

    def save_song_to_db(self, song_path):
        ml_author = get_model_with_insert(self.db_name, self.path)
        # The song's title is the path without the wav extension.
        save_song(song_path[:-4], ml_author, song_path)

    def predict(self):
        pass