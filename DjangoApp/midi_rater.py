from music21 import converter, instrument, note, chord, stream
from tqdm import tqdm
import os
import pandas as pd
from collections import Counter
import math
from music21.interval import Interval
from music21.scale import ConcreteScale, MajorScale
import numpy as np
from scipy.stats import mode

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

def extract_midis(path):

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
                if len(notes) > 0:
                    all_notes.append(notes)
                    all_note_offsets.append(note_offsets)
                    all_roots.append(roots)
                    if len(rests) <= 0:
                        rests = [0]
                        rest_durations = [0]
                    all_rests.append(rests)
                    all_rest_durations.append(rest_durations)


    return files, all_notes, all_rests, all_rest_durations, all_roots, all_note_offsets

def get_max_intervals(pitches):
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

def add_dicts(source, target):
    for key, value in source.items():
        if key in target.keys():
            target[key] += value
        else:
            target[key] = value

def get_music_df(path):
    files, pitches, rests, rest_durations, roots, note_offsets = extract_midis(path)
    all_music_data = []

    for i in tqdm(range(len(rest_durations))):
        # elements = []
        # for j in rest_durations[i]:
        #    elements.append(convert_to_float(get_number_from_duration(str(j))))

        rest_durations_local = list(map(convert_to_float, map(get_number_from_duration, map(str, rest_durations[i]))))
        note_durations_local = list(map(convert_to_float, get_duration_from_offset(note_offsets[i])))

        #max_interval = get_max_intervals(pitches[i])
        max_interval = 0

        intervals_count = Counter(get_direct_intervals(roots[i]))
        intervals_count = dict(sorted(intervals_count.items(), key=lambda item: item[1], reverse=True)[0:5])
        durations_count = Counter(note_durations_local)
        durations_count_merged = dict()
        for k, v in durations_count.items():
            k = round(k, 4)
            for used_key in durations_count_merged.keys():
                if math.isclose(used_key, k, rel_tol=1e-06):
                    durations_count_merged[used_key] += v
                    break
            else:
                durations_count_merged[k] = v
        durations_count = dict(sorted(durations_count_merged.items(), key=lambda item: item[1], reverse=True)[0:5])

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


    return pd.DataFrame(data=all_music_data, columns=['track_and_instrument', 'unique_pitches', 'unique_notes', 'notes_in_scale', 'notes_not_in_scale', 'max_interval', 'sum_rests', 'longest_rest', 'intervals_count', 'durations_count'])


def highest_sums(intervals):
    final_count_dict = dict()
    for interval_dict in intervals:
        for key, value in interval_dict.items():
            if key in final_count_dict.keys():
                final_count_dict[key] += value
            else:
                final_count_dict[key] = value
    top_5_count = dict(sorted(final_count_dict.items(), key=lambda item: item[1], reverse=True)[0:5])
    #print(final_count_dict)
    return list(top_5_count.keys())


def positionwise_mode(intervals):
    all_keys = []
    for interval_dict in intervals:
        keys_to_add = list(interval_dict.keys())
        while len(keys_to_add)<5:
            keys_to_add.append(None)
        all_keys.append(keys_to_add)

    all_keys = np.asarray(all_keys)
    counts = []
    for i in range(5):
        cnt = Counter(all_keys[:, i])
        counts.append(cnt)

    maximums = []
    for i in range(len(counts)):
        new_dict = dict()
        for idx, (key, value) in enumerate(sorted(counts[i].items(), key=lambda item: item[1], reverse=True)):
            if idx == 0:
                maximums.append(key)
            else:
                if key not in maximums and key:
                    new_dict[key] = value
        if i != 4:
            add_dicts(new_dict, counts[i + 1])
    #print(counts)
    return maximums


def positionwise_sums(intervals):
    all_items = []
    for interval_dict in intervals:
        items_to_add = list(interval_dict.items())
        while len(items_to_add)<5:
            items_to_add.append((0, 0))
        all_items.append(items_to_add)

    all_items = np.asarray(all_items)
    counts = []
    for i in range(5):
        column = all_items[:, i]
        col_dict = dict()
        for key, value in column:
            if key in col_dict.keys():
                col_dict[key] += value
            else:
                col_dict[key] = value
        counts.append(col_dict)

    maximums = []
    for i in range(len(counts)):
        new_dict = dict()
        for idx, (key, value) in enumerate(sorted(counts[i].items(), key=lambda item: item[1], reverse=True)):
            if idx == 0:
                maximums.append(key)
            else:
                if key not in maximums and key:
                    new_dict[key] = value
        if i != 4:
            add_dicts(new_dict, counts[i + 1])
    #print(counts)
    return maximums


def analyze_music(path, filter_outliers = False):
    df_music = get_music_df(path)
    if filter_outliers:
        mean_longest_rest = df_music['longest_rest'].mean()
        df_music = df_music[df_music['longest_rest']<mean_longest_rest]

    return df_music.drop(
        columns=['track_and_instrument']
    ).agg(
        {'unique_pitches':['mean', 'median'],
         'unique_notes':['mean', 'median'],
         'notes_in_scale':['mean', 'median'],
         'notes_not_in_scale':['mean', 'median'],
         'sum_rests':['mean', 'median'],
         'longest_rest':['mean', 'median'],
         'intervals_count':[highest_sums, positionwise_mode, positionwise_sums],
         'durations_count':[highest_sums, positionwise_mode, positionwise_sums]
         }
    )



if __name__ == '__main__':
    print('\n', analyze_music(r"D:/Egyetem/6.felev/Önlab/MarkovOutputs", True).to_string())
    print('\n', analyze_music(r"D:/MIDITest", True).to_string())