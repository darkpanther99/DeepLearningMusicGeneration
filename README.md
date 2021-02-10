# DeepLearningMusicGeneration
The project has 2 main parts:  
- Trying to generate Iron Maiden music from MIDI files  
- Trying to generate Classical rock/Blues rock music from continuous waveforms, such as mp3.  

I am going to use deep learning for the generation process, and because waveform and MIDI files differ from each other, I am storing them in different folders, like they are 2 separate projects.  
The first part of the project(s) is data visualization, so I made python scripts to parse and visualize MIDI and waveform music data.  

# Requirements
I am using Python 3.8.  
The required packages can be seen in the <a href="https://github.com/darkpanther99/DeepLearningMusicGeneration/blob/main/requirements.txt">requirements.txt</a> file.  
To download waveforms, I am using the Spotify API made for developers. In order to use my waveform scripts, you have to register for the spotify for developers program, and use your private authentication keys.
