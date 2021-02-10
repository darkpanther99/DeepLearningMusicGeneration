from music21 import converter, instrument, note, chord
import matplotlib.pyplot as plt

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

def parse_midi(path):
    chords = []
    durations = []

    midi = converter.parse(path)
    notes_to_parse = None

    notes_to_parse = midi.flat.notes

    #Lets make every note a chord!
    #Also save the durations
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            chords.append(chord.Chord([element]))
            durations.append(element.duration)
        elif isinstance(element, chord.Chord):
            chords.append(element)
            durations.append(element.duration)

    return chords, durations

def create_mapper(chords):
    pitchnames = sorted(set(str(item) for item in chords))
    mapper = dict((note, number) for number, note in enumerate(pitchnames))

    return mapper

def encode_using_mapper(chords, mapper):
    encodedsong=[]
    for c in chords:
        encodedsong.append(mapper[str(c)])

    return encodedsong


def decode_chords_using_mapper(numbers, mapper):
    outputnotes = []
    for number in numbers:
        outputnotes.append(chord_from_string(get_notes_from_chord(get_key_from_value(number, mapper))))

    return outputnotes

def combine_chords_with_durations(chords, durations):
    combined = []

    for i, j in zip(chords, durations):
        i = get_notes_from_chord(str(i))
        j = get_number_from_duration(str(j))
        combined.append(i + ';' + j)

    return combined


def visualize_notes_durations_separately(path):
    chords, durations = parse_midi(path)
    mapper = create_mapper(chords)
    durationmapper = create_mapper(durations)
    encodedsong = encode_using_mapper(chords, mapper)
    encodeddurations = encode_using_mapper(durations, durationmapper)

    plt.plot(encodedsong)
    plt.show()
    print(f'Song {path} containts {max(encodedsong)} unique notes.')
    plt.plot(encodeddurations)
    plt.show()
    print(f'Song {path} containts {max(encodeddurations)} unique durations.')


def visualize_note_durations_together(path):
    chords, durations = parse_midi(path)
    combined = combine_chords_with_durations(chords, durations)

    mapper = create_mapper(combined)
    encoded = encode_using_mapper(combined, mapper)

    plt.plot(encoded)
    plt.show()

    print(f'Song {path} containts {max(encoded)} unique combinations.')



visualize_notes_durations_separately("The_Trooper.mid")
visualize_notes_durations_separately("TrooperGuitar1.mid")

visualize_note_durations_together("The_Trooper.mid")
visualize_note_durations_together("TrooperGuitar1.mid")


