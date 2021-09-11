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

def model_song(request, model):
    print(model)
    return HttpResponse("OK")

def menu(request):
    return render(request, 'menu.html')

def listen(request):
    return render(request, 'listen.html')

def markov_chain(request):
    t = threading.Thread(target=create_markov)
    t.start()

    return HttpResponse("Markov Model is working in the background!")

def create_markov():
    MarkovModel()

def seed(request):
    #m = MLModel(name='MarkovChain')
    #m.save()

    return HttpResponse("Success!")

def try_song(request):
    #mc_author = MLModel.objects.filter(name="MarkovChain").first()
    #s = Song(title='test_markov', author=mc_author, path='output.wav')
    #s.save()
    s = Song.objects.filter(path='output.wav').first()

    variables = {
        'title' : s.title,
        'path' : s.path,
    }
    return render(request, 'song.html', variables)


def get_random_song(request):
    songs = Song.objects.all()
    chosen_song = random.choice(songs)
    variables = {
        'title': chosen_song.title,
        'author': chosen_song.author,
        'path': chosen_song.path,
    }
    return render(request, 'song.html', variables)


