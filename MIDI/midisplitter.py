from music21 import converter, note, chord
from music21.stream import Score
import music21
import os
import time
import random
from spoticonfig import LOCAL_ABSOLUTE_PATH

class MidiPart:
    def __init__(self, song, instrument, chords, durations):
        self.song = song
        self.instrument = instrument
        self.chords = chords
        self.durations = durations


path = LOCAL_ABSOLUTE_PATH #This is the folder on my PC which contains the Iron Maiden MIDIs

IRON_MAIDEN_INSTRUMENTS = ['Acoustic Guitar', 'Violoncello', 'Accordion', 'Piano', 'Sampler',
                           'StringInstrument', 'Xylophone', 'Electric Organ', 'Brass', 'Vibraphone',
                           'Electric Guitar', 'Viola', 'Electric Bass', 'Acoustic Bass', 'Voice', 'Celesta']

midiparts = []

start = time.time()

for file in os.listdir(path):
    print(file)
    midi = converter.parse(os.path.join(path, file))

    for part in midi.parts:
        chords=[]
        durations=[]
        for element in part.notesAndRests:
            if isinstance(element, note.Note):
                chords.append(chord.Chord([element]))
                durations.append(element.duration)
            elif isinstance(element, chord.Chord):
                chords.append(element)
                durations.append(element.duration)
            elif isinstance(element, note.Rest):
                chords.append(element)
                durations.append(element.duration)

        midiparts.append(MidiPart(file, part.partName, chords, durations))


end = time.time()

print(f'Parsing time:{end-start}')

TARGET_INSTRUMENT = "Electric Bass" #Choose one from IRON_MAIDEN_INSTRUMENTS

cnt=0
cntnotempty = 0
for i in midiparts:
    if i.instrument == TARGET_INSTRUMENT:
        cnt+=1
        if len(i.chords)>0:
            cntnotempty +=1

print()
#This counter shows the amount of songs, which have TARGET_INSTRUMENT in them
print(f'{cnt} songs have {TARGET_INSTRUMENT} in them')
#This counter shows the amount of songs, where the TARGET_INSTRUMENT track is not empty
#I don't know why a TARGET_INSTRUMENT track can be empty, when it has TARGET_INSTRUMENT in it, TODO figure it out!
print(f'{cntnotempty} songs have {TARGET_INSTRUMENT} in them where the track is not empty')

print()
#Here I print a random midipart object
rnd = random.randint(0,len(midiparts))
print(midiparts[rnd].song)
print(midiparts[rnd].instrument)
print(midiparts[rnd].chords)
print(midiparts[rnd].durations)

cntnotempty2 = 0
for i in midiparts:
    if len(i.chords)>0:
        cntnotempty2 +=1

print()
print(f'I have extracted {len(midiparts)} MIDI parts, from which {cntnotempty2} are not empty.')