from collections import Counter

from MLSongs.ml_agents.utilities import MidiPart
from music21 import converter, instrument, note, chord, stream
from tqdm import tqdm
import os

from MLSongs.ml_agents.utilities import most_frequent

def encode_notes(chords, mapper):
    encoded_data = []

    for c in chords:
        encoded = encode_using_mapper(c, mapper)
        encoded_data.append(encoded)

    return encoded_data

def get_chords_and_durations_of_instrument(midiparts, target_instrument):
    allchords = []
    alldurations = []

    for i in midiparts:
        if i.instrument == target_instrument:
            allchords.append(i.chords)
            alldurations.append(i.durations)

    return allchords, alldurations

def create_mapper(chords):
    pitchnames = sorted(set(str(item) for item in chords))
    mapper = dict((note, number) for number, note in enumerate(pitchnames))

    return mapper

def create_mapper_data(data):
    mapper_data = []
    for i in data:
        for j in i:
            mapper_data.append(j)

    return mapper_data

def clear_encoded_data(encoded_data, mapper):
    restkeysvalues = []
    for j in mapper.keys():
        if ('rest' in j):
            restkeysvalues.append(mapper[j])

    cleared_encoded_data = []

    for i in range(len(encoded_data)):
        if most_frequent(encoded_data[i]) not in restkeysvalues:
            cleared_encoded_data.append(encoded_data[i])

    return cleared_encoded_data

def filter_outliers(input, output, outlier_constant):

    outputcnt = Counter(output)
    outliers = []

    for i in outputcnt.keys():
        if outputcnt[i] < outlier_constant:
            outliers.append(i)

    assert (len(input) == len(output))

    newinput = []
    newoutput = []

    for i in range(len(output)):
        if (output[i] not in outliers):
            newinput.append(input[i])
            newoutput.append(output[i])

    input = newinput
    output = newoutput

    assert (len(input) == len(output))

    mapper_list = []  # Idx of the mapper list is the new value, the element is the old value.
    new_output_elements = set(output)

    for i in new_output_elements:
        mapper_list.append(i)

    newoutput = []

    for i in output:
        newoutput.append(mapper_list.index(i))

    output = newoutput

    return input, output, mapper_list

def make_slices(data, slice_length):
    for song in tqdm(data):
        if len(song) > slice_length:

            input = []
            output = []
            slice = []

            for idx, number in enumerate(song):
                if idx < slice_length:
                    slice.append(number)

            input.append(slice.copy())
            output.append(song[slice_length])

            # Sliding window
            for idx, number in enumerate(song):
                if idx >= slice_length and (idx + 1) < len(song):
                    slice.pop(0)
                    slice.append(number)
                    input.append(slice.copy())  # Copy is necessary, because of how pointers and lists work in Python
                    output.append(song[idx + 1])

    return input, output


def parse_everything_together(data, slice_length):
    notes = []
    input = []
    output = []
    slice = []

    for song in tqdm(data):
        for number in song:
            notes.append(number)

    for idx, note in tqdm(enumerate(notes)):
        if idx < slice_length:
            slice.append(number)

    input.append(slice.copy())
    output.append(notes[slice_length])

    # Sliding window
    for idx, number in tqdm(enumerate(notes)):
        if idx >= slice_length and (idx + 1) < len(notes):
            slice.pop(0)
            slice.append(number)
            input.append(slice.copy())  # Copy is necessary, because of how pointers and lists work in Python
            output.append(notes[idx + 1])

    return input, output



def encode_using_mapper(chords, mapper):
    encodedsong=[]
    for c in chords:
        encodedsong.append(mapper[str(c)])

    return encodedsong


def parse_midi_notes_and_durations(path):
    midiparts = []

    for file in tqdm(os.listdir(path)):
        midi = converter.parse(os.path.join(path, file))

        for part in midi.parts:
            chords = []
            durations = []
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

            if len(chords) > 0:
                midiparts.append(MidiPart(file, part.partName, chords, durations))
            else:
                for voice in part.voices:
                    chords = []
                    durations = []
                    for element in voice.notesAndRests:
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

    return midiparts