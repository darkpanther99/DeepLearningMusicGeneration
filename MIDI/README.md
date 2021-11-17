In midisplitter.py, I parse MIDI songs using the music21 library, and split them into separated parts, for example bass track, guitar track, etc.

In midivisualization.py, I parse MIDI songs, encode them into numbers, and plot them.

In output_to_midi.py, I convert numbers back into MIDI parts.

In markov_chain_baseline.py, I construct a Markov-chain to generate MIDI music using machine learning. I parse my MIDI-s using the midisplitter.py file's functions, encode them into numbers, construct a Markov-chain from a list made of those numbers, and generate 200 numbers, using the Markov-chain. After that I decode the numbers into Chord objects, and construct the generated MIDI file. This is kind of obsolete now, since the updated source code of the Markov chain is available in the [DjangoApp](https://github.com/darkpanther99/DeepLearningMusicGeneration/blob/main/DjangoApp/MLSongs/ml_agents/markov_chain.py) directory.

I have 7 directories where I train deep learning models to generate MIDI files. I implemented the models in Keras, by using Google Colab Notebooks. I download the data from my google drive and upload the outputted MIDIs there as well. To run the notebook properly, you have to supply it with data yourself.

In BaselineModel I train a basic stacked LSTM model to generate music. It can be run to generate music with durations embedded into the notes, or without durations, by using a constant tempo.

In MultiOutputModel I code the parsed MIDI files in a way that a separate piece of a neural network generates the notes, and the durations. It uses LSTMs like the Baseline Model. In the inference phase I generate the MIDI file from the separated notes and durations, coupling them together again.

MultiInstrumentModel is an upgraded version of the MultiOutputModel. It not only generated notes and durations, but it generates other instruments as well. It generates Guitar, Bass and Drums, however the Drums aren't working correctly when it is converted back to MIDI. They can't be played on the percussion MIDI channel, they are silent, but i don't know why. It is not because the neural network is not working correctly, the MIDI numbers are correct.

[MusicVAE](https://magenta.tensorflow.org/music-vae) is a whole different architecture. This is my take on Google Magenta's MusicVAE Autoencoder architecture to generate Iron Maiden guitar tracks. For the encoder part, it uses Bidirectional LSTM's to generate a latent code from the music and uses a hierarchical LSTM based decoder to make music from the latent code.

In Attention I train an attention based neural network, where I use self-attention instead of recurrent networks and also an LSTM-Attention architecture, in which the attention mechanism tries to learn the output of the recurrent network.

In Transformer I create a transformer, which is a bit more compley architecture that uses the self-attention mechanism. I was inspired by the [Music Transformer](https://magenta.tensorflow.org/music-transformer) architecture.

The GPT-2 Directory only contains a slightly modified notebook, which is a modification of [this](https://colab.research.google.com/github/sarthakmalik/GPT2.Training.Google.Colaboratory/blob/master/Train_a_GPT_2_Text_Generating_Model_w_GPU.ipynb). There I finetune a GPT-2 to generate music instead of text.