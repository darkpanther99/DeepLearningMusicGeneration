from keras.models import load_model
from MLSongs.database.models import MLModel, Song


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
        ml_author = MLModel.objects.filter(name=self.db_name).first()

        if not ml_author:
            ml_author = MLModel(name=self.db_name, path=self.path)
            ml_author.save()

        self.model = load_model(ml_author.path)

    def save_song_to_db(self, song_path):
        # Since I only have 1 LSTMModel record, that will be the MLModel.
        ml_author = MLModel.objects.filter(name=self.db_name).first()

        if not ml_author:
            ml_author = MLModel(name=self.db_name, path = self.path)
            ml_author.save()

        # The song's title is the path without the wav extension.
        s = Song(title=song_path[:-4], author=ml_author, path=song_path)
        s.save()

    def predict(self):
        pass