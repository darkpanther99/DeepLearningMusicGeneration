from midisplitter import MidiPart, generate_midi_parts, generate_midi_parts_without_chords
from midivisualization import create_mapper, encode_using_mapper
from output_to_midi import create_midi_without_chords, create_midi_without_durations, create_midi_with_durations
import numpy as np
import mchmm as mc
import random
from music21 import instrument

def get_key_from_value(value, dict):
    return list(dict.keys())[list(dict.values()).index(value)]

if __name__=='__main__':
    midiparts = generate_midi_parts()

    TARGET_INSTRUMENT = 'Electric Guitar' #Choose one from IRON_MAIDEN_INSTRUMENTS

    allchords = []
    alldurations = []

    for i in midiparts:
        if i.instrument == TARGET_INSTRUMENT:
            allchords.append(i.chords)
            alldurations.append(i.durations)

    chord_mapper_data = []
    for i in allchords:
        for j in i:
            chord_mapper_data.append(j)
    mapper = create_mapper(chord_mapper_data)

    duration_mapper_data = []
    for i in alldurations:
        for j in i:
            duration_mapper_data.append(j)
    duration_mapper = create_mapper(duration_mapper_data)

    encoded_chords = []
    durationsdata = []

    for c in allchords:
        encoded = encode_using_mapper(c, mapper)
        for i in encoded:
            encoded_chords.append(i)

    for d in alldurations:
        encoded = encode_using_mapper(d, duration_mapper)
        for i in encoded:
            durationsdata.append(i)


    encoded_chord_string = []
    encoded_duration_string = []
    for i in encoded_chords:
        encoded_chord_string.append(str(i))

    for i in durationsdata:
        encoded_duration_string.append(str(i))

    chordsmc = mc.MarkovChain().from_data(encoded_chord_string)
    #These are the transition matrices. I am leaving these prints in for debug purposes
    #print(chordsmc.observed_matrix)
    #print(chordsmc.observed_p_matrix)

    durationmc = mc.MarkovChain().from_data(encoded_duration_string)

    for j in range(0, 10):
        chord_states = None
        duration_states = None
        #This is an ugly hack, don't try anything like this at home.
        #chordsmc.simulate() throws an exception 90% of the time, because of floating point rounding errors, because my transition matrix is too huge.
        #A workaround is this ugly while True loop, which will spin the CPU until we get an errorless simulation.
        while chord_states is None:
            try:
                ids, chord_states = chordsmc.simulate(200, tf = np.asarray(chordsmc.observed_matrix).astype('float64'), start=encoded_chord_string[random.randint(0, len(encoded_chord_string))])
            except:
                pass

        while duration_states is None:
            try:
                durids, duration_states = durationmc.simulate(200, tf=np.asarray(durationmc.observed_matrix).astype('float64'), start=encoded_duration_string[random.randint(0, len(encoded_duration_string))])
            except:
                pass

        music = []
        musicdurations = []

        for i in chord_states:
            note = get_key_from_value(int(i), mapper)
            music.append(note)

        for i in duration_states:
            duration = get_key_from_value(int(i), duration_mapper)
            musicdurations.append(duration)

        print(music)
        print(musicdurations)

        #create_midi_without_durations(music, instrument.ElectricBass(), f'MarkovOutputs\Maiden_Bass_{j}.mid')
        create_midi_with_durations(music, musicdurations, instrument.ElectricGuitar(), f'MarkovOutputs\Maiden_Guitar_{j}.mid')

