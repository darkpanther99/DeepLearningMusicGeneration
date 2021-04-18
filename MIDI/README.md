In midisplitter.py, I parse MIDI songs using the music21 library, and split them into separated parts, for example bass track, guitar track, etc.

In midivisualization.py, I parse MIDI songs, encode them into numbers, and plot them.

In output_to_midi.py, I convert numbers back into MIDI parts.

In markov_chain_baseline.py, I construct a Markov-chain to generate MIDI music using machine learning. I parse my MIDI-s using the midisplitter.py file's functions, encode them into numbers, construct a Markov-chain from a list made of those numbers, and generate 200 numbers, using the Markov-chain. After that I decode the numbers into Chord objects, and construct the generated MIDI file.

I also included 2 MIDI files the user can play around with. One is a MIDI file of Iron Maiden's song The Trooper, the other is a separated Guitar track of that song.

I have 4 directories where I train deep learning models to generate MIDI files. I implemented the models in Keras, by using Google Colab Notebooks. I download the data from my google drive and upload the outputted MIDIs there as well. To run the notebook properly, you have to supply it with data yourself.

In BaselineModel I train a basic stacked LSTM model to generate music. It can be run to generate music with durations embedded into the notes, or without durations, by using a constant tempo.

In MultiOutputModel I code the parsed MIDI files in a way that a separate piece of a neural network generates the notes, and the durations. It uses LSTMs like the Baseline Model. In the inference phase I generate the MIDI file from the separated notes and durations, coupling them together again.

MultiInstrumentModel is an upgraded version of the MultiOutputModel. It not only generated notes and durations, but it generates other instruments as well. It generates Guitar, Bass and Drums, however the Drums aren't working correctly when it is converted back to MIDI. They can't be played on the percussion MIDI channel, they are silent, but i don't know why. It is not because the neural network is not working correctly, the MIDI numbers are correct.

[MusicVAE](https://magenta.tensorflow.org/music-vae) is a whole different architecture. This is my take on Google Magenta's MusicVAE Autoencoder architecture to generate Iron Maiden guitar tracks. For the encoder part, it uses Bidirectional LSTM's to generate a latent code from the music and uses a hierarchical LSTM based decoder to make music from the latent code.
