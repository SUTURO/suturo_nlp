# SUTURO NLP
This repository contains the natural language generation and processing components used by the SUTURO bachelor and master projects.

## Provided Packages

### suturo_julius
Contains dictionaries, julius configuration files and launchfiles.
It's only purpose is to start julius.
### suturo_nlg
Contains the nlg_ros node. This node interfaces with different python3 scripts producing sentences.
Also contains the knowledge_interface. This node takes does rosprolog queries to answer basic questions.
### suturo_textparser
contains the sling parser,its ros interfacing node and a textparser thats simplyfiies and reduces the julius output.
### suturo_speechparser
Just contains some launchfiles starting node from all the above packages.
### activate_language_processing
contains package used for Receptionist task