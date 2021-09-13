from django.http import HttpResponse
from django.shortcuts import render
from MLSongs.database.models import MLModel, Song
from rest_framework import viewsets
from MLSongs.serializers import SongSerializer
from MLSongs.ml_agents.markov_chain import MarkovModel
from MLSongs.ml_agents.LSTM import LSTMModel
from MLSongs.ml_agents.GPT_2 import Music_GPT_2
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
    ML_model_name = ""

    if "markov" in model.lower():
        ML_model_name = "MarkovChain"
    elif "lstm" in model.lower():
        ML_model_name = "LSTMModel"
    elif "gpt" in model.lower():
        ML_model_name = "GPT-2Model"


    mc_author = MLModel.objects.filter(name=ML_model_name).first()
    if not mc_author:
        return HttpResponse(f"No ML model found for {ML_model_name}. Try executing it, or a different model.")
    try:
        chosen_song = random.choice(Song.objects.filter(author=mc_author).all())
    except IndexError:
        return HttpResponse(f"No generated songs found for {ML_model_name}. Execute the model ang generate some!")


    variables = {
        'title': chosen_song.title,
        'author': chosen_song.author,
        'path': chosen_song.path,
    }
    return render(request, 'song.html', variables)

def execute_model(request, model, count):
    print(model)

    if "markov" in model.lower():
        t = threading.Thread(target=create_markov, args=(count,))
        t.start()
        return HttpResponse("Markov Model is working in the background!")

    if "lstm" in model.lower():
        t = threading.Thread(target=create_LSTM, args=(count,))
        t.start()
        return HttpResponse("LSTM is working in the background!")

    if "gpt" in model.lower():
        t = threading.Thread(target=create_gpt)
        t.start()
        return HttpResponse("GPT-2 postprocessor is working in the background!")

    return HttpResponse("OK")

def create_gpt():
    gpt = Music_GPT_2()
    data = gpt.load_data()
    clean_data = gpt.preprocess_data(data)
    gpt.predict(clean_data, -1)


def create_markov(count):
    mc = MarkovModel()
    data = mc.load_data()
    chords, durations = mc.preprocess_data(data)
    print("Building model")
    mc.build_model(chords, durations)
    print("Generating music")
    mc.generate_music(chords, durations, count)

def create_LSTM(count):
    lstm = LSTMModel()
    data = lstm.load_data()
    input = lstm.preprocess_data(data)
    lstm.build_model()
    lstm.predict(input, count, 0.8)

def seed(request):
    #m = MLModel(name='MarkovChain')
    #m.save()

    return HttpResponse("Success!")





