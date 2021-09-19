from tqdm import tqdm
from music21 import converter, instrument, note, chord, stream

class MidiPart:
    def __init__(self, song, instrument, chords, durations):
        self.song = song
        self.instrument = instrument
        self.chords = chords
        self.durations = durations


def get_key_from_value(value, dict):
    return list(dict.keys())[list(dict.values()).index(value)]

def most_frequent(paramlist):
    # https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    counter = 0
    num = paramlist[0]

    for i in paramlist:
        curr_frequency = paramlist.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def get_notes_from_chord(chord):
    if chord.startswith("<music21.chord.Chord "):
        chord = chord[len("<music21.chord.Chord "):]
    if chord.endswith(">"):
        chord = chord[:-1]
    chord = chord.replace(" ", ",")
    return chord


def get_number_from_duration(duration):
    if duration.startswith("<music21.duration.Duration "):
        duration = duration[len("<music21.duration.Duration "):]
    if duration.endswith(">"):
        duration = duration[:-1]
    duration = duration.replace(" ", ",")
    return duration


def combine_chords_with_durations(chords, durations):
    combined = []

    for i, j in zip(chords, durations):
        i = get_notes_from_chord(str(i))
        j = get_number_from_duration(str(j))
        combined.append(i + ';' + j)

    return combined


def chord_from_string(chordstring):
    notes = chordstring.split(";")
    return chord.Chord(notes)


def convert_to_float(frac_str):
    # From: https://stackoverflow.com/questions/1806278/convert-fraction-to-float
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac
