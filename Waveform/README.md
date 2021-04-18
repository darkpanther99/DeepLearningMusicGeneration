In this directory you can find scripts and notebooks about making music consisting of continuous waveforms.

To download waveforms, I am using the Spotify API made for developers. In order to use my waveform downloading scripts, you have to register for the spotify for developers program, and use your private authentication keys, because I will not be making those(hidden in my config file) public.

In spotiscript.py, I use the Spotify API to get preview URL-s from a few artists, from which I can download songs to use them for training. This is currently unused, because I am not using a dataset downloaded from spotify, I am using the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset instead.

In datavisualization.py, I get the urls to a dataset using my spotify script, download the first song from it, parse it using librosa and plot it using matplotlib, both in time domain, and in the frequency domain(using logarithmic scale).

In the WavenetBaseline directory, I have a jupyter notebook, called WavenetBaseline.ipynb, where I parse 12 Schubert pieces from the MAESTRO dataset, μ-law encode the parsed data, and feed it to a Wavenet neural network model using a Keras DataGenerator.
After the training, I autoregressively generate music using the model, and a softmax resampling function with a temperature parameter to inject some more randomness into it.
However, currently even with the resampling(which I find strange) it can only generate a straight line as music(the same note forever), this might be because I am doing something wrong, or because Colab doesn't have enough resources to train a Wavenet. Specifically a better GPU would be needed to be able to train it in time. On a Tesla T4 GPU, with the implemented architecture it takes about 85 hours to train 1 epoch on the full dataset.