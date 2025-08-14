#!/bin/bash

# Create a parent directory for the virtual environments
VENV_PARENT_DIR="venvs2"
mkdir -p "$VENV_PARENT_DIR"

# Define paths to the virtual environments
VENV1_DIR="$VENV_PARENT_DIR/whisper_venv"
VENV2_DIR="$VENV_PARENT_DIR/rasa_venv"

# Create virtual environments
python3 -m venv "$VENV1_DIR"
python3 -m venv "$VENV2_DIR"

pip install --upgrade pip
pip install rospy==1.17.0

##################################
# VENV 1 INSTALLATION
##################################
source "$VENV1_DIR/bin/activate"
echo "Installing packages in $VENV1_DIR..."
pip install --upgrade pip
pip install \
    inflect==0.2.5 \
    levenshtein==0.25.1 \
    librosa==0.11.0 \
    Metaphone==0.6 \
    nltk==3.9.1 \
    noisereduce==3.0.3 \
    numpy==1.24.4 \
    openai-whisper==20240930 \
    pathlib==1.0.1 \
    PyAudio==0.2.14 \
    PyYAML==6.0.2 \
    regex==2024.9.11 \
    requests==2.32.3 \
    rospkg==1.5.1 \
    setuptools==44.0.0 \
    soundfile==0.12.1 \
    spacy==3.7.5 \
    SpeechRecognition==3.10.4 \
    torch==2.4.1 \
    urllib3==2.2.3 \
    word2number==1.1   
deactivate

##################################
# VENV 2 INSTALLATION
##################################
source "$VENV2_DIR/bin/activate"
echo "Installing packages in $VENV2_DIR..."
pip install --upgrade pip
pip install \
    keras==2.12.0 \
    rasa==3.6.20 \
    rasa-sdk==3.6.2 \
    redis==4.6.0 \
    regex==2022.10.31 \
    requests==2.32.3 \
    requests-oauthlib==2.0.0 \
    requests-toolbelt==1.0.0 \
    rocketchat-API==1.30.0 \
    tensorboard==2.12.3 \
    tensorboard-data-server==0.7.2 \
    tensorflow==2.12.0 \
    tensorflow-estimator==2.12.0 \
    tensorflow-hub==0.13.0 \
    tensorflow-io-gcs-filesystem==0.32.0 \
    tensorflow-text==2.12.0 
deactivate

echo "✅ Both virtual environments are set up inside '$VENV_PARENT_DIR'!"
