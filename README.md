# sf-ha-onsetdetection

![](img/diagram.png)  

Notebooks:
mc_crop : After putting audio files into "musiccritic" folder, run this. Cropped .wav files together with altered onset annotations will go into "musiccritic_cropped" folder. It crops the unwanted (silence, somebody talking etc.) parts from beginning and end of the recordings.    

main: First cell installs GuitarSet and unzips to "guitarset" folder. Usage of onset detection and audio player is shown.    
chord_segmentation: In chord exercises, rise time of the chords and spacing of individual strings are important. This notebook shows a prototype.    
od_comparison: Onset detection algorithms applied on whole datasets. You need "madmom" library for CNN Onset Detector.  


GuitarSet: https://github.com/marl/GuitarSet  
DFT, Peak Detection taken from  https://github.com/MTG/sms-tools  
CNN Onset Detector: https://github.com/CPJKU/madmom  
Annotation of the music critic dataset is done with https://github.com/srvrc/Sound-Annotator    
  
To be able to run the audio player (quickPlayer.py), you need to install PyAudio. This is necessary for observing the results.  
<code>$ sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 </code>  
<code>$ sudo apt-get install ffmpeg libav-tools </code>  
<code>$ sudo pip install pyaudio </code>  


GuitarSet

|                     | F-score | Precision | Recall |
|---------------------|---------|-----------|--------|
| CNN Onset Detector  | 0.84   | 0.78     | 0.92  |
| GuitarOnsetDetector |   0.71      |    0.95       |    0.59  |
| SF-HA               | 0.84   | 0.89     | 0.81  |



Music Critic Dataset 

|                     | F-score | Precision | Recall |
|---------------------|---------|-----------|--------|
| CNN Onset Detector  | 0.70   | 0.59     | 0.92  |
| GuitarOnsetDetector |   0.80 | 0.80    | 0.80  |
| SF-HA               | 0.85   | 0.86     | 0.84  |





                                 GuitarSet                    Music Critic Dataset
| Chord Files         | F-score | Precision | Recall | - | F-score | Precision | Recall |
|---------------------|---------|-----------|--------| - |---------|-----------|--------|
| CNN Onset Detector  | 0.82   | 0.78     | 0.88     |   | 0.59    | 0.46      | 0.93   | 
| GuitarOnsetDetector | 0.69   | 0.95     | 0.56     |   | 0.74    | 0.74      | 0.74   |
| SF-HA               | 0.81   | 0.91     | 0.76     |   | 0.84    | 0.84      | 0.85   |


                                 GuitarSet                    Music Critic Dataset
| Solo Files         | F-score | Precision | Recall | -  | F-score | Precision | Recall |
|---------------------|---------|-----------|--------| - |---------|-----------|--------|
| CNN Onset Detector  | 0.86   | 0.79     | 0.95          || 0.78     | 0.69  | 0.92 | 
| GuitarOnsetDetector |   0.73      |  0.95  |    0.60  ||  0.85      |    0.86  |    0.84  |
| SF-HA               | 0.86   | 0.88     | 0.86          || 0.85 |  0.88     | 0.84  |
