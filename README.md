# SUTURO NLP
This repository contains the natural language generation and processing components used by the SUTURO bachelor and master projects.

## Provided Packages

### activate_language_processing
This package is utilized for the RoboCup @Home Receptionist task. The script is triggered by Planning to detect names and favorite drinks of guests conversing with the robot, finally providing a list/response to Knowledge and Planning. TODO: rename to a more appropriate name.

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

