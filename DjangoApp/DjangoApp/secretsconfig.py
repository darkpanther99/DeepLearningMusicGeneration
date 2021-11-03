import os


SECRETS = {
    "DJANGO_SECRET_KEY" : "TODO" #insert your secret key here
}

LOCAL_ABSOLUTE_PATH = os.path.join(os.getcwd(), "training_midis")
SOUNDFONT_PATH = os.path.join(os.getcwd(), "soundfonts", "default.sf2")