from django.http import HttpResponse
from django.shortcuts import render
import random
import threading
from MLSongs.database.db_services import get_model, get_songs_by_author, get_all_songs, create_empty_song
from MLSongs.ml_agents.execute_models import create_markov, create_LSTM, create_multi_lstm, create_attention,\
    create_transformer, create_multi_transformer, create_gpt, create_vae


def get_random_song(request):
    songs = get_all_songs()
    try:
        chosen_song = random.choice(songs)
    except IndexError:
        return HttpResponse(f"No generated songs found. Execute a model and generate some!")
    variables = {
        'title': chosen_song.title,
        'author': chosen_song.author,
        'path': chosen_song.path,
    }
    return render(request, 'song.html', variables)


def model_song(request, model, instrument):

    chosen_song = create_empty_song()
    ML_model_name = ""

    if "markov" in model.lower():
        if "guitar" in instrument.lower():
            ML_model_name = "MarkovGuitar"
        elif "bass" in instrument.lower():
            ML_model_name = "MarkovBass"
    elif "lstm" in model.lower():
        if "guitar" in instrument.lower():
            ML_model_name = "LSTMModel"
        elif "bass" in instrument.lower():
            ML_model_name = "LSTMBassModel"
        elif "multi" in instrument.lower():
            ML_model_name = 'LSTMMultiInstrumentModel'
    elif "gpt" in model.lower():
        if "guitar" in instrument.lower():
            ML_model_name = "GPT-2Model"
    elif "vae" in model.lower():
        if "guitar" in instrument.lower():
            ML_model_name = "MusicVAEGuitar"
        elif "bass" in instrument.lower():
            ML_model_name = "MusicVAEBass"
    elif "attention" in model.lower():
        if "guitar" in instrument.lower():
            ML_model_name = "AttentionModel"
        elif "bass" in instrument.lower():
            ML_model_name = "AttentionModelBass"
    elif "transformer" in model.lower():
        if "guitar" in instrument.lower():
            ML_model_name = "TransformerModel"
        elif "bass" in instrument.lower():
            ML_model_name = "TransformerModelBass"
        elif "multi" in instrument.lower():
            ML_model_name = 'TransformerMultiInstrumentModel'

    ml_author = get_model(ML_model_name)
    if not ml_author:
        return HttpResponse(f"No ML model found for {ML_model_name}. Try executing it, or a different model.")
    try:
        chosen_song = random.choice(get_songs_by_author(ML_model_name))
    except IndexError:
        return HttpResponse(f"No generated songs found for {ML_model_name}. Execute the model and generate some!")

    variables = {
        'title': chosen_song.title,
        'author': chosen_song.author,
        'path': chosen_song.path,
    }
    return render(request, 'song.html', variables)


def execute_model(request, model, instrument, count, temp):

    if "markov" in model.lower():
        if "guitar" in instrument.lower() or "bass" in instrument.lower():
            t = threading.Thread(target=create_markov, args=(count, instrument))
            t.start()
            return HttpResponse("Markov Model is working in the background!")

    if "lstm" in model.lower():
        if "guitar" in instrument.lower() or "bass" in instrument.lower():
            t = threading.Thread(target=create_LSTM, args=(count, instrument, temp))
            t.start()
            return HttpResponse("LSTM is working in the background!")
        elif "multi" in instrument.lower():
            t = threading.Thread(target=create_multi_lstm, args=(count, temp))
            t.start()
            return HttpResponse("LSTM is working in the background!")

    if "attention" in model.lower():
        if "guitar" in instrument.lower() or "bass" in instrument.lower():
            t = threading.Thread(target=create_attention, args=(count, instrument, temp))
            t.start()
            return HttpResponse("Attention based model is working in the background!")

    if "transformer" in model.lower():
        if "guitar" in instrument.lower() or "bass" in instrument.lower():
            t = threading.Thread(target=create_transformer, args=(count, instrument, temp))
            t.start()
            return HttpResponse("Transformer based model is working in the background!")
        elif "multi" in instrument.lower():
            t = threading.Thread(target=create_multi_transformer, args=(count, temp))
            t.start()
            return HttpResponse("Transformer based model is working in the background!")

    if "gpt" in model.lower():
        if "guitar" in instrument.lower():
            t = threading.Thread(target=create_gpt)
            t.start()
            return HttpResponse("GPT-2 postprocessor is working in the background!")

    if "vae" in model.lower():
        if "guitar" in instrument.lower() or "bass" in instrument.lower():
            t = threading.Thread(target=create_vae, args=(count, instrument, temp))
            t.start()
            return HttpResponse("MusicVAE is working in the background!")

    return HttpResponse("No model found with that name and instrument combination!")


def debug(request):
    temp = 0.9
    count = 5
    instruments = ['guitar', 'bass']
    for instrument in instruments:
        create_markov(count, instrument)
        create_LSTM(count, instrument, temp)
        create_vae(count, instrument, temp)
        create_attention(count, instrument, temp)
        create_transformer(count, instrument, temp)
    return HttpResponse('OK')


def execute_model_once(request, model, instrument):
    return execute_model_without_temp(request, model, instrument, 1)


def execute_model_without_temp(request, model, instrument, count):
    return execute_model(request, model, instrument, count, 0.8)


def help(request):
    return render(request, 'help.html')


def about(request):
    return render(request, 'about.html')