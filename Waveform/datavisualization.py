from spotiscript import download_dataset
import matplotlib.pyplot as plt
import numpy as np
import librosa
import urllib


songs = download_dataset()
print(f'We have {len(songs)} audio samples, which are 30 second long each.')

urllib.request.urlretrieve(songs[0], 'temp.mp3')
temppath='temp.mp3'

#I used CD quality sampling rate (44100 Hz)
#With this sampling rate, a 30 second sample of a song consists of 1 323 000 floats, which is a lot.
song, sr = librosa.load(temppath, sr=44100)

plt.plot(song)
plt.show()

#TODO find max frequency of songs, so I can reduce the sampling rate accordingly, 44kHz is too much
plt.specgram(song)
plt.show()


