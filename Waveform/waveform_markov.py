from spotiscript import download_dataset
from tqdm import tqdm
import numpy as np
import librosa
from scipy.io import wavfile
import urllib
import mchmm as mc
import random


'''
Note: This was a failed experiment. Markov-chains can't generate musical waveform this way.
'''
#7 bits MU-LAW
def mulaw_encode(samples, divider):
    # Rescale to -1.0..1.0. Encode to -64..63. Return 0..127.
    return (librosa.mu_compress(samples / divider, mu = 127, quantize=True) + 64).astype('uint8')


def mulaw_decode(samples, multiplier):
    # Rescale from 0..127 to -64..63. Decode to -1.0..1.0. Multiply by the scaling factor
    return (((librosa.mu_expand(samples.astype('int16') - 64, mu = 127, quantize=True) ).astype('float16')) * multiplier)


dataset = download_dataset()

songs_array = []
SONG_COUNT = 2

for i in tqdm(range(SONG_COUNT)):

    urllib.request.urlretrieve(dataset[i], 'temp.mp3')
    temppath='temp.mp3'

    song, sr = librosa.load(temppath, sr=2*8192)

    for j in song:
        songs_array.append(j)

MAX_CONST = max(songs_array)
mulaw_songs = mulaw_encode(songs_array, MAX_CONST)
#print(max(songs_array))

mulaw_song_string = []
for i in mulaw_songs:
    mulaw_song_string.append(str(i))

generator = mc.MarkovChain().from_data(mulaw_song_string)

print("simulation starting!")

states = None
while states is None:
    try:
        ids, states = generator.simulate(16*8192, tf = np.asarray(generator.observed_matrix).astype('float64'), start=mulaw_song_string[random.randint(0, len(mulaw_song_string))])
    except:
        pass


music = mulaw_decode(states, MAX_CONST)

print(len(music))
print(max(music))
print(music)

#wavfile.write("test_output.wav", rate = 2*8192, data = music.astype('float32'))

