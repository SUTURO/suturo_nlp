# SUTURO NLP
This repository contains the natural language generation and processing components used by the SUTURO bachelor and master projects.

## Provided Packages

### activate_language_processing
This package is utilized for the RoboCup @Home Receptionist task. The script is triggered by Planning to detect names and favorite drinks of guests conversing with the robot, finally providing a list/response to Knowledge and Planning. For more information, please visit the [suturo_nlp](https://github.com/SUTURO/SUTURO-documentation/wiki/suturo_nlp) wiki for more information.

Here are the required packages for the packages:
```
inflect==0.2.5 
levenshtein==0.25.1 
librosa==0.11.0 
Metaphone==0.6 
nltk==3.9.1 
noisereduce==3.0.3 
numpy==1.24.4 
openai-whisper==20240930 
pathlib==1.0.1 
PyAudio==0.2.14 
PyYAML==6.0.2 
regex==2024.9.11 
requests==2.32.3 
rospkg==1.5.1 
setuptools==44.0.0 
soundfile==0.12.1 
spacy==3.7.5 
SpeechRecognition==3.10.4 
torch==2.4.1 
urllib3==2.2.3 
word2number==1.1  
```

## Obsolete Packages

Store older more-or-less finished pieces of code. Will be removed at some point in the future.

### audio_compare
Contains the scripts used to recognize a doorbell from a .wav or microphone input.

### pos_recognizer
Scripts for recognizing parts of speech in commands used in the EGPSR

### suturo_nlg
**(no longer in use)**
Contains the nlg_ros node. This node interfaces with different python3 scripts producing sentences. Also contains the knowledge_interface. This node uses rosprolog queries to answer basic questions.

### suturo_textparser
**(no longer in use)**
Contains the sling parser, its ros interfacing node and a text parser that simplifies and reduces the julius output.

### test_noisy_input
A small and simple tool to test speech recognition while playing a recording of a loud and noisy environment. Parameters can be adjusted to test and improve detection quality.

