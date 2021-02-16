from spotiscript import download_dataset
import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import specshow
import urllib


songs = download_dataset()
print(f'We have {len(songs)} audio samples, which are 30 second long each.')

urllib.request.urlretrieve(songs[0], 'temp.mp3')
temppath='temp.mp3'


song, sr = librosa.load(temppath, sr=2*8192)

plt.plot(song)
plt.show()


D = librosa.stft(song)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure()
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()



