from MLSongs.database.models import Song, MLModel


def get_songs_by_author(author_name):
    author_object = MLModel.objects.filter(name=author_name).first()
    return Song.objects.filter(author = author_object).all()

def get_all_songs():
    return Song.objects.all()

def get_model(ML_model_name):
    return MLModel.objects.filter(name=ML_model_name).first()

def get_model_with_insert(db_name, path):
    ml_author = MLModel.objects.filter(name=db_name).first()

    if not ml_author:
        ml_author = MLModel(name=db_name, path=path)
        ml_author.save()

    return ml_author

def save_song(title, author, path):
    s = Song(title=title, author=author, path=path)
    s.save()

def create_empty_song():
    return Song(author = MLModel())

