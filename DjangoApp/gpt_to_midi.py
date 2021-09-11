from MLSongs.ml_agents.utilities import create_midi_with_embedded_durations

files = ["samples-201.txt", "samples-401.txt", "samples-601.txt", "samples-801.txt",]


for file in files:
    f = open(file, "rt")
    gpt_output = f.read()[27:]
    f.close()
    notes = gpt_output.split(" ")[1:-2] #cut away the first and last notes, sicne they might be corrupted
    filename = file[:-3] + "mid"
    create_midi_with_embedded_durations(notes, filename = filename)
