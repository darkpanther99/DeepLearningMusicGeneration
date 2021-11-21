# DeepLearningMusicGeneration

The project had 2 main parts:  
- To generate Iron Maiden music from MIDI files  
- To generate Classical(or other instrumental) music from continuous waveforms, such as mp3.  

I used machine learning, to be more precise, mostly deep learning for the generation process, and because waveform and MIDI files differ from each other, I am storing them in different folders, like they are 2 separate projects.  
Unfortunately the continuous waveforms project couldn't succeed, because it would have needed more processing power, that I don't have. I hope I can go back to it in the future!  
Because of this, I put all my efforts into the MIDI project.  

# Running the code

## Training

In the MIDI and Waveform folders, I have jupyter notebooks with the model training code in them. Supply it with data (I used Google Colab, and its Drive integration for this), run the appropriate cells, and the training can begin!

## Inference / Generating music

The project was tested on Python 3.8 and 3.9.  
I haven't tried running it on earlier versions, and at the time I'm writing this, TensorFlow doesn't support Python 3.10.  
1. Have [FluidSynth](https://www.fluidsynth.org/) on your computer added to PATH to be able to synthesize music from MIDIs into Waveforms.
2. Clone the repository.
3. Install the required Python packages.
```
pip install -r requirements.txt
```
4. Using the command line, navigate to the DjangoApp folder of the repository.
5. (Optional) Write your Django secret key (which can be generated [here](https://djecrety.ir/)) to the appropriate line of the [config file](https://github.com/darkpanther99/DeepLearningMusicGeneration/blob/main/DjangoApp/DjangoApp/secretsconfig.py).
6. Run the following commands to set the database up and run the Web App:
```
python manage.py migrate
python manage.py makemigrations MLSongs
python manage.py migrate MLSongs
python manage.py runserver
```
7. Using a web browser, naviagte to the URL on which the server is running (the output of the runserver command contains it).
8. If you need help with using the web app, go to the help page.
9. Enjoy your generated music!


# Results

The MIDI project's results are the trained models [here](https://github.com/darkpanther99/DeepLearningMusicGeneration/tree/main/DjangoApp/ml_models). To generate music using them, use the web application I made.
