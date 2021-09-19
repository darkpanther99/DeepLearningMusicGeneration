import numpy as np
from MLSongs.ml_agents.utilities import get_key_from_value, chord_from_string, get_notes_from_chord, get_number_from_duration, convert_to_float
from music21 import converter, instrument, note, chord, stream
import os
from midi2audio import FluidSynth
from DjangoApp.secretsconfig import SOUNDFONT_PATH


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = np.squeeze(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def change_midi_instrument(midi_path, new_instrument):
    s = converter.parse(midi_path)
    for p in s.parts:
        p.insert(0, new_instrument)
    s.write('midi', midi_path)


def midi_to_wav(midi_path, wav_path, delete_midi = False):
    fs = FluidSynth(sound_font=SOUNDFONT_PATH)
    fs.midi_to_audio(midi_path, wav_path)
    if delete_midi:
        os.remove(midi_path)


def decode_chords_using_mapper(numbers, mapper):
    outputnotes = []
    for number in numbers:
        outputnotes.append(chord_from_string(get_notes_from_chord(get_key_from_value(number, mapper))))

    return outputnotes


def generate_multi_instrument_notes(model, starting_slice, starting_duration, starting_bass_slice, starting_drum_slice,
                                    mapper, duration_mapper, bass_mapper, drum_mapper, mapperlist=None, duration_mapper_list=None, temp=1.0, duration_temp=0.8,
                                    bass_temp=0.8, drum_temp=0.8):
    """ Generate notes from the neural network based on a sequence of notes """
    pattern = starting_slice
    duration_pattern = starting_duration
    bass_pattern = starting_bass_slice
    drum_pattern = starting_drum_slice
    prediction_output_notes = []
    prediction_output_durations = []
    prediction_output = []
    prediction_output_bass = []
    prediction_output_bass_ret = []
    prediction_output_drum = []
    prediction_output_drum_ret = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_duration_input = np.reshape(duration_pattern, (1, len(duration_pattern), 1))
        prediction_bass_input = np.reshape(bass_pattern, (1, len(bass_pattern), 1))
        prediction_drum_input = np.reshape(drum_pattern, (1, len(drum_pattern), 1))

        note_prediction, duration_prediction, bass_prediction, drum_prediction = model.predict(
            {"notes_in": prediction_input, "durations_in": prediction_duration_input, "bass_in": prediction_bass_input,
             "drum_in": prediction_drum_input}
        )

        # prediction = sample(prediction, temp)
        index = sample(note_prediction, temp)
        duration_index = sample(duration_prediction, duration_temp)
        bass_index = sample(bass_prediction, bass_temp)
        drum_index = sample(drum_prediction, drum_temp)

        # index = np.argmax(prediction)
        if mapperlist is not None:  # Idx of the mapper list is the new value, the element is the old value. This is used when I filter for outliers.
            index = mapperlist[index]

        if duration_mapper_list is not None:
            duration_index = duration_mapper_list[duration_index]

        result = get_key_from_value(index, mapper)
        prediction_output_notes.append(result)

        duration_result = get_key_from_value(duration_index, duration_mapper)
        prediction_output_durations.append(duration_result)

        bass_result = get_key_from_value(bass_index, bass_mapper)
        prediction_output_bass.append(bass_result)

        drum_result = get_key_from_value(drum_index, drum_mapper)
        prediction_output_drum.append(drum_result)

        pattern = np.append(pattern, index / float(len(mapper)))
        duration_pattern = np.append(duration_pattern, duration_index / float(len(duration_mapper)))
        bass_pattern = np.append(bass_pattern, bass_index / float(len(bass_mapper)))
        drum_pattern = np.append(drum_pattern, drum_index / float(len(drum_mapper)))

        pattern = pattern[1:len(pattern)]
        duration_pattern = duration_pattern[1:len(duration_pattern)]
        bass_pattern = bass_pattern[1:len(bass_pattern)]
        drum_pattern = drum_pattern[1:len(drum_pattern)]

    for i in range(500):
        prediction_output.append(str(prediction_output_notes[i]) + ';' + str(prediction_output_durations[i]))
        prediction_output_bass_ret.append(str(prediction_output_bass[i]) + ';' + str(prediction_output_durations[i]))
        prediction_output_drum_ret.append(str(prediction_output_drum[i]) + ';' + str(prediction_output_durations[i]))

    return prediction_output, prediction_output_bass_ret, prediction_output_drum_ret


from music21.stream import Part

def create_drum_part_with_durations(prediction_output):
    offset = 0
    output_notes = []

    from music21.midi.percussion import PercussionMapper

    pm = PercussionMapper()

    # create note and chord objects based on the values generated by the model
    for i in range(len(prediction_output)):
        pattern = prediction_output[i]
        splitpattern = pattern.split(";")
        pattern = splitpattern[0]

        duration = get_number_from_duration(splitpattern[1])
        # pattern is a chord
        if ('chord' in pattern):
            notes = []
            pattern = get_notes_from_chord(pattern)
            patternpitches = pattern.split(',')
            for current_note in patternpitches:
                new_note = note.Note(current_note)
                try:
                    new_note.storedInstrument = pm.midiPitchToInstrument(new_note.pitch)
                except:
                    new_note.storedInstrument = instrument.Percussion()
                notes.append(new_note)
                new_note.offset = offset
                output_notes.append(new_note)
        # pattern is a rest
        elif ('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            try:
                new_note.storedInstrument = pm.midiPitchToInstrument(new_note.pitch)
            except:
                new_note.storedInstrument = instrument.Percussion()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += convert_to_float(duration)

    midipart = stream.Part(output_notes)

    return midipart

def create_midipart_with_durations(prediction_output, target_instrument=instrument.Piano()):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for i in range(len(prediction_output)):
        pattern = prediction_output[i]
        splitpattern = pattern.split(";")
        pattern = splitpattern[0]

        duration = get_number_from_duration(splitpattern[1])
        # pattern is a chord
        if ('chord' in pattern):
            notes = []
            pattern = get_notes_from_chord(pattern)
            patternpitches = pattern.split(',')
            for current_note in patternpitches:
                new_note = note.Note(current_note)
                new_note.storedInstrument = target_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a rest
        elif ('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = target_instrument  # ???
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = target_instrument
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += convert_to_float(duration)

    midipart = stream.Part(output_notes)

    return midipart

def generate_notes(model, network_input, mapper, mapperlist = None, temp=1.0):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(prediction_input, verbose=0)

        #prediction = sample(prediction, temp)
        index = sample(prediction, temp)

        #index = np.argmax(prediction)
        if mapperlist is not None: #Idx of the mapper list is the new value, the element is the old value. This is used when I filter for outliers.
            index=mapperlist[index]

        result = get_key_from_value(index, mapper)
        prediction_output.append(result)

        pattern = np.append(pattern, index/float(len(mapper)))

        pattern = pattern[1:len(pattern)]

    return prediction_output

# Source: https://github.com/alexissa32/DataScienceMusic
def create_midi_without_chords(prediction_output, target_instrument=instrument.Piano(), filename='test_output.mid'):
    '''
    First step:
    Only notes, no chords
    Static 4/4 beat
    Rests
    '''
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a rest
        if ('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = target_instrument  # ???
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = target_instrument
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=filename)


def create_midi_without_durations(prediction_output, target_instrument=instrument.Piano(), filename='test_output.mid'):
    '''
    Second step:
    Chords and notes
    Static 4/4 beat
    Rests
    '''
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('chord' in pattern):
            notes = []
            pattern = get_notes_from_chord(pattern)
            patternpitches = pattern.split(',')
            for current_note in patternpitches:
                new_note = note.Note(current_note)
                new_note.storedInstrument = target_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a rest
        elif ('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = target_instrument  # ???
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = target_instrument
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=filename)


def create_midi_with_durations(prediction_output, output_durations, target_instrument=instrument.Piano(),
                               filename='test_output.mid'):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for i in range(len(prediction_output)):
        pattern = prediction_output[i]
        duration = get_number_from_duration(output_durations[i])
        # pattern is a chord
        if ('chord' in pattern):
            notes = []
            pattern = get_notes_from_chord(pattern)
            patternpitches = pattern.split(',')
            for current_note in patternpitches:
                new_note = note.Note(current_note)
                new_note.storedInstrument = target_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a rest
        elif ('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = target_instrument  # ???
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = target_instrument
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += convert_to_float(duration)

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=filename)


def create_midi_with_embedded_durations(prediction_output, target_instrument=instrument.Piano(),
                                        filename='test_output.mid'):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for i in range(len(prediction_output)):
        pattern = prediction_output[i]
        splitpattern = pattern.split(";")
        pattern = splitpattern[0]

        duration = get_number_from_duration(splitpattern[1])
        # pattern is a rest
        if ('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = target_instrument  # ???
            output_notes.append(new_rest)
        # pattern is a chord
        elif (',' in pattern):
            notes = []
            pattern = get_notes_from_chord(pattern)
            patternpitches = pattern.split(',')
            for current_note in patternpitches:
                new_note = note.Note(current_note)
                new_note.storedInstrument = target_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = target_instrument
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += convert_to_float(duration)

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=filename)