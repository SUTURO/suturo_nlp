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
    catkin-pkg==1.0.0 \
    certifi==2024.8.30 \
    cffi==1.17.1 \
    charset-normalizer==3.4.0 \
    distro==1.9.0 \
    docutils==0.20.1 \
    filelock==3.16.1 \
    fsspec==2024.10.0 \
    idna==3.10 \
    importlib-metadata==8.5.0 \
    inflection==0.5.1 \
    jinja2==3.1.4 \
    joblib==1.4.2 \
    llvmlite==0.41.1 \
    MarkupSafe==2.1.5 \
    more-itertools==10.5.0 \
    mpmath==1.3.0 \
    networkx==3.1 \
    nltk==3.9.1 \
    noisereduce==3.0.3 \
    numba==0.58.1 \
    numpy==1.24.4 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nccl-cu12==2.20.5 \
    nvidia-nvjitlink-cu12==12.6.77 \
    nvidia-nvtx-cu12==12.1.105 \
    openai-whisper==20240930 \
    pathlib==1.0.1 \
    pip==20.0.2 \
    pkg-resources==0.0.0 \
    PyAudio==0.2.14 \
    pycparser==2.22 \
    pyparsing==3.1.4 \
    python-dateutil==2.9.0.post0 \
    PyYAML==6.0.2 \
    regex==2024.9.11 \
    requests==2.32.3 \
    rospkg==1.5.1 \
    setuptools==44.0.0 \
    six==1.16.0 \
    soundfile==0.12.1 \
    spacy==3.7.5 \
    SpeechRecognition==3.10.4 \
    sympy==1.13.3 \
    tiktoken==0.7.0 \
    torch==2.4.1 \
    tqdm==4.66.6 \
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
    absl-py==1.4.0 \
    aio-pika==8.2.3 \
    aiofiles==24.1.0 \
    aiogram==2.15 \
    aiohttp==3.9.5 \
    aiohttp-retry==2.9.0 \
    aiormq==6.4.2 \
    aiosignal==1.3.1 \
    APScheduler==3.9.1.post1 \
    astunparse==1.6.3 \
    async-timeout==4.0.3 \
    attrs==22.1.0 \
    Babel==2.9.1 \
    backports.zoneinfo==0.2.1 \
    bidict==0.23.1 \
    boto3==1.35.53 \
    botocore==1.35.53 \
    CacheControl==0.12.14 \
    cachetools==5.5.0 \
    certifi==2024.8.30 \
    cffi==1.17.1 \
    charset-normalizer==3.4.0 \
    click==8.1.7 \
    cloudpickle==2.2.1 \
    colorclass==2.2.2 \
    coloredlogs==15.0.1 \
    colorhash==1.2.1 \
    confluent-kafka==2.6.0 \
    cryptography==43.0.3 \
    cycler==0.12.1 \
    dask==2022.10.2 \
    dnspython==2.3.0 \
    docopt==0.6.2 \
    fbmessenger==6.0.0 \
    fire==0.7.0 \
    flatbuffers==24.3.25 \
    fonttools==4.54.1 \
    frozenlist==1.5.0 \
    fsspec==2024.10.0 \
    future==1.0.0 \
    gast==0.4.0 \
    google-auth==2.35.0 \
    google-auth-oauthlib==1.0.0 \
    google-pasta==0.2.0 \
    greenlet==3.1.1 \
    grpcio==1.67.1 \
    h11==0.14.0 \
    h5py==3.11.0 \
    httptools==0.6.4 \
    humanfriendly==10.0 \
    idna==3.10 \
    importlib_metadata==8.5.0 \
    importlib_resources==6.4.5 \
    jax==0.4.13 \
    jmespath==1.0.1 \
    joblib==1.2.0 \
    jsonpickle==3.0.4 \
    jsonschema==4.17.3 \
    keras==2.12.0 \
    kiwisolver==1.4.7 \
    libclang==18.1.1 \
    locket==1.0.0 \
    magic-filter==1.0.12 \
    Markdown==3.7 \
    MarkupSafe==2.1.5 \
    matplotlib==3.5.3 \
    mattermostwrapper==2.2 \
    ml-dtypes==0.2.0 \
    msgpack==1.1.0 \
    multidict==5.2.0 \
    networkx==2.6.3 \
    nltk==3.9.1 \
    numpy==1.23.5 \
    oauthlib==3.2.2 \
    opt_einsum==3.4.0 \
    packaging==20.9 \
    pamqp==3.2.1 \
    partd==1.4.1 \
    pillow==10.4.0 \
    pip==24.3.1 \
    pkg_resources==0.0.0 \
    pkgutil_resolve_name==1.3.10 \
    pluggy==1.5.0 \
    portalocker==2.10.1 \
    prompt-toolkit==3.0.28 \
    propcache==0.2.0 \
    protobuf==4.23.3 \
    psycopg2-binary==2.9.10 \
    pyasn1==0.6.1 \
    pyasn1_modules==0.4.1 \
    pycparser==2.22 \
    pydantic==1.10.9 \
    pydot==1.4.2 \
    PyJWT==2.9.0 \
    pykwalify==1.8.0 \
    pymongo==4.3.3 \
    pyparsing==3.1.4 \
    pyrsistent==0.20.0 \
    python-crfsuite==0.9.11 \
    python-dateutil==2.8.2 \
    python-engineio==4.10.1 \
    python-socketio==5.11.4 \
    pytz==2022.7.1 \
    PyYAML==6.0.2 \
    questionary==1.10.0 \
    randomname==0.1.5 \
    rasa==3.6.20 \
    rasa-sdk==3.6.2 \
    redis==4.6.0 \
    regex==2022.10.31 \
    requests==2.32.3 \
    requests-oauthlib==2.0.0 \
    requests-toolbelt==1.0.0 \
    rocketchat-API==1.30.0 \
    rsa==4.9 \
    ruamel.yaml==0.17.21 \
    ruamel.yaml.clib==0.2.8 \
    s3transfer==0.10.3 \
    sanic==21.12.2 \
    Sanic-Cors==2.0.1 \
    sanic-jwt==1.8.0 \
    sanic-routing==0.7.2 \
    scikit-learn==1.1.3 \
    scipy==1.10.1 \
    sentry-sdk==1.14.0 \
    setuptools==69.5.1 \
    simple-websocket==1.1.0 \
    six==1.16.0 \
    sklearn-crfsuite==0.3.6 \
    slack_sdk==3.33.3 \
    SQLAlchemy==1.4.54 \
    structlog==23.3.0 \
    structlog-sentry==2.1.0 \
    tabulate==0.9.0 \
    tarsafe==0.0.4 \
    tensorboard==2.12.3 \
    tensorboard-data-server==0.7.2 \
    tensorflow==2.12.0 \
    tensorflow-estimator==2.12.0 \
    tensorflow-hub==0.13.0 \
    tensorflow-io-gcs-filesystem==0.32.0 \
    tensorflow-text==2.12.0 \
    termcolor==2.4.0 \
    terminaltables==3.1.10 \
    threadpoolctl==3.5.0 \
    toolz==1.0.0 \
    tqdm==4.66.6 \
    twilio==8.2.2 \
    typing_extensions==4.12.2 \
    typing-utils==0.1.0 \
    tzlocal==5.2 \
    ujson==5.10.0 \
    urllib3==1.26.20 \
    uvloop==0.21.0 \
    wcwidth==0.2.13 \
    webexteamssdk==1.6.1 \
    websockets==10.4 \
    Werkzeug==3.0.6 \
    wheel==0.44.0 \
    wrapt==1.14.1 \
    wsproto==1.2.0 \
    yarl==1.15.2 
deactivate

echo "✅ Both virtual environments are set up inside '$VENV_PARENT_DIR'!"
