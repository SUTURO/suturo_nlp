#!/bin/bash

# This script should build the NLP workspace properly.

set -e

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo "This script must NOT be run as root"
    exit 1
fi

# ----- Get user input for workspace directories -----

read -p "Where should the workspaces be built? (Default: $HOME/ros): " WS_PATH

# Trim whitespace
WS_PATH="$(echo "$WS_PATH" | xargs)"
# Default
WS_PATH="${WS_PATH:-"$HOME/ros"}"

if [[ $WS_PATH != /* ]]; then
    WS_PATH="$PWD/$WS_PATH"
fi

WS_PATH="$(realpath -m "$WS_PATH")"

read -p "Is this path correct? '$WS_PATH' [y|n]: " CONFIRM
# lower
CONFIRM="${CONFIRM,,}"

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "yes" ]]; then
    echo "Path not confirmed. Script cancelled."
    exit 1
fi

echo "Workspace path: $WS_PATH"

# ----- Environment Path Input -----

read -p "Where is your environment directory placed? (Default: $HOME/.virtualenvs): " ENV_PATH

ENV_PATH="$(echo "$ENV_PATH" | xargs)"

# If default
USED_DEFAULT=false
if [[ -z "$ENV_PATH" ]]; then
    ENV_PATH="$HOME/.virtualenvs"
    USED_DEFAULT=true
fi

if [[ $ENV_PATH != /* ]]; then
    ENV_PATH="$PWD/$ENV_PATH"
fi

ENV_PATH="$(realpath -m "$ENV_PATH")"

read -p "Is this path correct? '$ENV_PATH' [y|n]: " ENV_CONFIRM

ENV_CONFIRM="${ENV_CONFIRM,,}"

if [[ "$ENV_CONFIRM" != "y" && "$ENV_CONFIRM" != "yes" ]]; then
    echo "Path not confirmed. Script cancelled."
    exit 1
fi

echo "Environment path: $ENV_PATH"

# Create default directory if it does not exist
if [[ ! -d "$ENV_PATH" ]]; then
    if [[ "$USED_DEFAULT" == true ]]; then
        echo "Default environment directory does not exist. Creating it..."
        mkdir -p "$ENV_PATH"
    else
        echo "Environment directory not found. Please create it or choose another one."
        exit 1
    fi
fi

export WORKON_HOME="$ENV_PATH"
export PROJECT_HOME="$HOME/Projects"

# ----- Setup virtualenvwrapper -----

# Check if WORKON variables are set in .bashrc
# Append variables in .bashrc if not found
if ! grep -q "^export WORKON_HOME=" "$HOME/.bashrc"; then
    echo "export WORKON_HOME=$WORKON_HOME" >> "$HOME/.bashrc"
fi

if ! grep -q "^export PROJECT_HOME=" "$HOME/.bashrc"; then
    echo "export PROJECT_HOME=$HOME/Projects" >> "$HOME/.bashrc"
fi

# ----- Before we create workspaces we download all dependencies -----

# Update system and install necessary dependencies
echo "Installing necessary dependencies..."

sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y \
    python3.10 \
    python3.10-dev \
    python3.12 \
    python3.12-dev \
    virtualenvwrapper \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg

echo "Done."

# Check if virtualenvwrapper can be found
if [ -f "/usr/share/virtualenvwrapper/virtualenvwrapper.sh" ]; then
  # Source virtualenvwrapper
  source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
else
    echo "virtualenvwrapper.sh not found"
    exit 1
fi

# Add virtualenvwrapper source to bashrc if not included
if ! grep -q "^source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" "$HOME/.bashrc"; then
  echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> "$HOME/.bashrc"
fi

# ----- Create envs -----

# We have to use virtualenv here, because mkvirtualenv starts a interactive shell
# that breaks this script.

echo "Creating virtual environments..."

# --- rasa_venv ---
if [ ! -d "$WORKON_HOME/rasa_venv" ]; then
    virtualenv -p python3.10 "$WORKON_HOME/rasa_venv" --system-site-packages
else
    echo "rasa_venv already exists"
fi

# --- whisper_venv ---
if [ ! -d "$WORKON_HOME/whisper_venv" ]; then
    virtualenv -p python3.12 "$WORKON_HOME/whisper_venv" --system-site-packages
else
    echo "whisper_venv already exists"
fi

echo "Done."

# ----- Install dependencies for venvs -----

# Install dependencies for each virtualenv
echo "Installing dependencies for each environment..."

source "$WORKON_HOME/whisper_venv/bin/activate"
# workon whisper_venv

python3.12 -m pip install --upgrade pip setuptools wheel

python3.12 -m pip install \
    inflect \
    levenshtein \
    librosa \
    Metaphone \
    nltk \
    noisereduce \
    numpy \
    openai-whisper \
    pathlib \
    PyAudio \
    PyYAML \
    regex \
    requests \
    rospkg \
    setuptools \
    soundfile \
    spacy \
    SpeechRecognition \
    torch \
    urllib3 \
    word2number \
    coverage==7.12.0

deactivate

source "$WORKON_HOME/rasa_venv/bin/activate"
# workon rasa_venv

python3.10 -m pip install --upgrade pip setuptools wheel

python3.10 -m pip install  \
    nltk \
    keras \
    rasa \
    rasa-sdk \
    redis \
    regex \
    requests \
    requests-oauthlib \
    requests-toolbelt \
    rocketchat-API \
    tensorboard \
    tensorboard-data-server \
    tensorflow \
    tensorflow-estimator \
    tensorflow-hub \
    tensorflow-io-gcs-filesystem \
    tensorflow-text \
    cffi

# should fix the Pillow issue if you try to train rasa model
python3.10 -m pip install --no-cache-dir -I Pillow

deactivate

echo "All dependencies installed!"

# ----- Creating workspaces -----

echo "Creating workspaces..."

mkdir -p "$WS_PATH/messages_ws/src/"
echo "Created messages_ws in $WS_PATH" 
mkdir -p "$WS_PATH/nlp_ws/src/"
echo "Created nlp_ws in $WS_PATH"

# Clone repositories

cd "$WS_PATH/messages_ws/src/"

# Check if repositories already exists
if [[ ! -d suturo_nlp_msgs ]]; then
    git clone git@github.com:SUTURO/suturo_nlp_msgs.git
else
    echo "suturo_nlp_msgs already exists"
fi

cd "$WS_PATH/nlp_ws/src"

# Check if repositories already exists
if [[ ! -d suturo_nlp ]]; then
    git clone git@github.com:SUTURO/suturo_nlp.git
else
    echo "suturo_nlp already exists"
fi

if [[ ! -d suturo_rasa ]]; then
    git clone git@github.com:SUTURO/suturo_rasa.git
else
    echo "suturo_rasa already exists"
fi

echo "Done."

# Building

source /opt/ros/jazzy/setup.bash

cd "$WS_PATH/messages_ws/"
colcon build

cd "$WS_PATH/nlp_ws/"
# build twice because sometimes there are some cmake warnings
for i in {1..2}; do
    colcon build
done

# Add nlp workspace source to bashrc
if ! grep -q "^source $WS_PATH/nlp_ws/install/setup.bash" "$HOME/.bashrc"; then
  echo "source $WS_PATH/nlp_ws/install/setup.bash" >> "$HOME/.bashrc"
fi

if ! grep -q "^source $WS_PATH/messages_ws/install/setup.bash" "$HOME/.bashrc"; then
  echo "source $WS_PATH/messages_ws/install/setup.bash" >> "$HOME/.bashrc"
fi

echo "Workspaces built!"
echo ""
echo "For further proceeding read the wiki in: https://github.com/SUTURO/SUTURO-documentation/wiki/NLP-Home"
echo ""
echo "Don't forget to edit your .bashrc file and add any missing source components." 