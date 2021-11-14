from MLSongs.ml_agents.multi_instrument_LSTM import MultiInstrumentLSTM
from MLSongs.ml_agents.markov_chain import MarkovModel
from MLSongs.ml_agents.LSTM import LSTMModel
from MLSongs.ml_agents.GPT_2 import Music_GPT_2
from MLSongs.ml_agents.MusicVAE import MusicVAE
from MLSongs.ml_agents.Attention import AttentionModel
from MLSongs.ml_agents.Transformer import TransformerModel
from MLSongs.ml_agents.multi_instrument_transformer import MultiInstrumentTransformer

def create_multi_transformer(count, temp):
    multi_transformer = MultiInstrumentTransformer()
    data = multi_transformer.load_data()
    multi_transformer.preprocess_data(data)
    multi_transformer.build_model()
    multi_transformer.predict(data, count, temp, 250)
    print("Generation task has finished!")

def create_transformer(count, instrument, temp):
    transformer = TransformerModel(instrument)
    data = transformer.load_data()
    data = transformer.preprocess_data(data)
    transformer.build_model()
    transformer.predict(data, count, temp, 250)
    print("Generation task has finished!")

def create_attention(count, instrument, temp):
    att = AttentionModel(instrument)
    data = att.load_data()
    data = att.preprocess_data(data)
    att.build_model()
    att.predict(data, count, temp, 250)
    print("Generation task has finished!")

def create_vae(count, instrument, temp):
    vae = MusicVAE(instrument)
    data = vae.load_data()
    data = vae.preprocess_data(data)
    vae.build_model()
    vae.predict(data, count, temp)
    print("Generation task has finished!")

def create_gpt():
    gpt = Music_GPT_2()
    data = gpt.load_data()
    clean_data = gpt.preprocess_data(data)
    gpt.predict(clean_data, -1)
    print("Generation task has finished!")

def create_multi_lstm(count, temp):
    multi_lstm = MultiInstrumentLSTM()
    data = multi_lstm.load_data()
    guitar_input, durations_input, bass_input, drum_input = multi_lstm.preprocess_data(data)
    multi_lstm.build_model()
    multi_lstm.predict(data, count, temp)
    print("Generation task has finished!")

def create_markov(count, instrument):
    mc = MarkovModel(instrument)
    data = mc.load_data()
    chords, durations = mc.preprocess_data(data)
    mc.build_model(chords, durations)
    mc.generate_music(chords, durations, count, 250)
    print("Generation task has finished!")

def create_LSTM(count, instrument, temp):
    lstm = LSTMModel(instrument)
    data = lstm.load_data()
    data = lstm.preprocess_data(data)
    lstm.build_model()
    lstm.predict(data, count, temp, 250)
    print("Generation task has finished!")