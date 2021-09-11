from django.http import HttpResponse
from django.shortcuts import render
from MLSongs.database.models import MLModel, Song
from rest_framework import viewsets
from MLSongs.serializers import SongSerializer
from MLSongs.ml_agents.markov_chain import MarkovModel
import random
import threading

class SongViewSet(viewsets.ModelViewSet):
    queryset = Song.objects.all()
    serializer_class = SongSerializer

def get_random_song(request):
    songs = Song.objects.all()
    chosen_song = random.choice(songs)
    variables = {
        'title': chosen_song.title,
        'author': chosen_song.author,
        'path': chosen_song.path,
    }
    return render(request, 'song.html', variables)

def model_song(request, model):
    print(model)

    chosen_song = Song(author = MLModel())

    if "markov" in model.lower():
        mc_author = MLModel.objects.filter(name="MarkovChain").first()
        chosen_song = random.choice(Song.objects.filter(author=mc_author).all())

    variables = {
        'title': chosen_song.title,
        'author': chosen_song.author,
        'path': chosen_song.path,
    }
    return render(request, 'song.html', variables)

def execute_model(request, model):
    print(model)

    if "markov" in model.lower():
        t = threading.Thread(target=create_markov)
        t.start()
        return HttpResponse("Markov Model is working in the background!")

    return HttpResponse("OK")

def create_markov():
    MarkovModel()

def seed(request):
    #m = MLModel(name='MarkovChain')
    #m.save()

    return HttpResponse("Success!")





