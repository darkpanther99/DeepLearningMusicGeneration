In midisplitter.py, I parse MIDI songs using the music21 library, and split them into separated parts, for example bass track, guitar track, etc.

In midivisualization.py, I parse MIDI songs, encode them into numbers, and plot them.

In output_to_midi.py, I convert numbers back into MIDI parts.

In markov_chain_baseline.py, I construct a Markov-chain to generate MIDI music using machine learning. I parse my MIDI-s using the midisplitter.py file's functions, encode them into numbers, construct a Markov-chain from a list made of those numbers, and generate 200 numbers, using the Markov-chain. After that I decode the numbers into Chord objects, and construct the generated MIDI file.

I also included 2 MIDI files the user can play around with. One is a MIDI file of Iron Maiden's song The Trooper, the other is a separated Guitar track of that song.
