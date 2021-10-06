from music21 import converter, instrument, note, chord, stream
from tqdm import tqdm
import os
import pandas as pd
from collections import Counter

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


def get_notes_from_pitch(chord):
    if chord.startswith("<music21.pitch.Pitch "):
        chord = chord[len("<music21.pitch.Pitch "):]
    if chord.endswith(">"):
        chord = chord[:-1]
    return chord

def get_notes_from_note(chord):
    if chord.startswith("<music21.note.Note "):
        chord = chord[len("<music21.note.Note "):]
    if chord.endswith(">"):
        chord = chord[:-1]
    return chord

def get_number_from_duration(duration):
    if duration.startswith("<music21.duration.Duration "):
        duration = duration[len("<music21.duration.Duration "):]
    if duration.endswith(">"):
        duration = duration[:-1]
    duration = duration.replace(" ", ",")
    return duration

def get_durations(path):

    files = []
    all_notes = []
    all_rests = []
    all_rest_durations = []
    all_roots = []
    all_note_offsets = []

    for file in tqdm(os.listdir(path)):
        if file.endswith("mid"):
            midi = converter.parse(os.path.join(path, file))

            for part in midi.parts:
                rests = []
                notes = []
                roots = []
                note_offsets = []
                rest_durations = []
                for element in part.notesAndRests:
                    if isinstance(element, note.Note):
                        notes.append(element.pitch)
                        roots.append(element.pitch)
                        note_offsets.append(element.offset)
                    elif isinstance(element, chord.Chord):
                        roots.append(element.root())
                        for n in element.pitches:
                            notes.append(n)
                        note_offsets.append(element.offset)
                    elif isinstance(element, note.Rest):
                        rests.append(element)
                        rest_durations.append(element.duration)

                files.append(file+" "+str(part.partName))
                all_notes.append(notes)
                all_note_offsets.append(note_offsets)
                all_roots.append(roots)
                if len(rests) <= 0:
                    rests = [0]
                    rest_durations = [0]
                all_rests.append(rests)
                all_rest_durations.append(rest_durations)


    return files, all_notes, all_rests, all_rest_durations, all_roots, all_note_offsets

from music21.interval import Interval
from music21.scale import ConcreteScale, MajorScale

def get_intervals(pitches):
    lengths = []
    for pitch in pitches:
        for pitch2 in pitches:
            lengths.append(Interval(pitch, pitch2).semitones)

    max_len = max(lengths)
    abs_min = abs(min(lengths))
    if abs_min > max_len:
        max_len = abs_min

    return max_len

def get_direct_intervals(roots):
    direct_intervals = []

    for i in range(len(roots)-1):
        direct_intervals.append(Interval(roots[i], roots[i+1]).semitones)

    return direct_intervals

def get_duration_from_offset(offsets):
    durations = []

    for i in range(len(offsets) - 1):
        durations.append(offsets[i + 1] - offsets[i])

    return durations

def analyze_music(path):
    files, pitches, rests, rest_durations, roots, note_offsets = get_durations(path)
    all_music_data = []

    for i in tqdm(range(len(rest_durations))):
        # elements = []
        # for j in rest_durations[i]:
        #    elements.append(convert_to_float(get_number_from_duration(str(j))))

        rest_durations_local = list(map(convert_to_float, map(get_number_from_duration, map(str, rest_durations[i]))))
        note_durations_local = get_duration_from_offset(note_offsets[i])

        max_interval = get_intervals(pitches[i])

        intervals_count = Counter(get_direct_intervals(roots[i]))
        durations_count = Counter(note_durations_local)

        notes = list(map(note.Note, pitches[i]))
        notes_str = list(map(str, notes))
        notes_str = list(map(get_notes_from_note, notes_str))
        unique_notes_str = list(set(notes_str))
        unique_notes_cnt = len(unique_notes_str)

        # song_scale = ConcreteScale(pitches=unique_notes_str)
        major = MajorScale()
        scale_diff = major.deriveRanked(unique_notes_str)
        notes_in_scale = scale_diff[0][0]
        notes_not_in_scale = unique_notes_cnt - notes_in_scale

        pitches_str = list(map(str, pitches[i]))
        pitches_str = list(map(get_notes_from_pitch, pitches_str))
        unique_pitches = list(set(pitches_str))

        music_data = (files[i], len(unique_pitches), unique_notes_cnt, notes_in_scale, notes_not_in_scale, max_interval,
                sum(rest_durations_local), max(rest_durations_local), intervals_count, durations_count)
        all_music_data.append(music_data)
        print(music_data)


    return pd.DataFrame(data=all_music_data, columns=['track_and_instrument', 'unique_pitches', 'unique_notes', 'notes_in_scale', 'notes_not_in_scale', 'max_interval', 'sum_rests', 'longest_rest', 'intervals_count', 'durations_count'])

if __name__ == '__main__':
    analyze_music(r"D:/Egyetem/6.felev/Önlab/MarkovOutputs")