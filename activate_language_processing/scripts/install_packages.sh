#!/bin/bash

# Exit if any command fails
set -e

# Optional: create and activate a virtual environment
# python3 -m venv venv
# source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install all packages with specific versions
pip install \
  accelerate==1.7.0 \
  annotated-types==0.7.0 \
  anyio==4.5.2 \
  audioread==3.0.1 \
  blis==0.7.11 \
  catalogue==2.0.10 \
  catkin-pkg==1.0.0 \
  certifi==2024.8.30 \
  cffi==1.17.1 \
  charset-normalizer==3.4.0 \
  click==8.1.8 \
  cloudpathlib==0.20.0 \
  confection==0.1.5 \
  contourpy==1.1.1 \
  cycler==0.12.1 \
  cymem==2.0.11 \
  decorator==5.2.1 \
  distro==1.9.0 \
  docutils==0.20.1 \
  en-core-web-sm==3.7.1 \
  exceptiongroup==1.3.0 \
  filelock==3.16.1 \
  fonttools==4.57.0 \
  fsspec==2024.10.0 \
  h11==0.16.0 \
  httpcore==1.0.9 \
  httpx==0.28.1 \
  huggingface-hub==0.31.2 \
  idna==3.10 \
  importlib_metadata==8.5.0 \
  importlib_resources==6.4.5 \
  inflection==0.5.1 \
  Jinja2==3.1.4 \
  joblib==1.4.2 \
  kiwisolver==1.4.7 \
  langcodes==3.4.1 \
  language_data==1.3.0 \
  lazy_loader==0.4 \
  librosa==0.11.0 \
  llvmlite==0.41.1 \
  marisa-trie==1.2.1 \
  markdown-it-py==3.0.0 \
  MarkupSafe==2.1.5 \
  matplotlib==3.7.5 \
  mdurl==0.1.2 \
  more-itertools==10.5.0 \
  mpmath==1.3.0 \
  msgpack==1.1.0 \
  murmurhash==1.0.12 \
  networkx==3.1 \
  nltk==3.9.1 \
  noisereduce==3.0.3 \
  numba==0.58.1 \
  numpy==1.24.4 \
  nvidia-cublas-cu12==12.6.4.1 \
  nvidia-cuda-cupti-cu12==12.6.80 \
  nvidia-cuda-nvrtc-cu12==12.6.77 \
  nvidia-cuda-runtime-cu12==12.6.77 \
  nvidia-cudnn-cu12==9.5.1.17 \
  nvidia-cufft-cu12==11.3.0.4 \
  nvidia-cufile-cu12==1.11.1.6 \
  nvidia-curand-cu12==10.3.7.77 \
  nvidia-cusolver-cu12==11.7.1.2 \
  nvidia-cusparse-cu12==12.5.4.2 \
  nvidia-cusparselt-cu12==0.6.3 \
  nvidia-nccl-cu12==2.26.2 \
  nvidia-nvjitlink-cu12==12.6.85 \
  nvidia-nvtx-cu12==12.6.77 \
  ollama==0.4.8 \
  openai-whisper==20240930 \
  packaging==25.0 \
  pathlib==1.0.1 \
  pillow==10.4.0 \
  pip==25.1.1 \
  platformdirs==4.3.8 \
  pooch==1.8.2 \
  preshed==3.0.9 \
  psutil==7.0.0 \
  PyAudio==0.2.14 \
  pycparser==2.22 \
  pydantic==2.10.6 \
  pydantic_core==2.27.2 \
  Pygments==2.19.1 \
  pyparsing==3.1.4 \
  python-dateutil==2.9.0.post0 \
  PyYAML==6.0.2 \
  regex==2024.9.11 \
  requests==2.32.3 \
  rich==14.0.0 \
  rospkg==1.5.1 \
  safetensors==0.5.3 \
  scikit-learn==1.6.1 \
  scipy==1.10.1 \
  setuptools==44.0.0 \
  shellingham==1.5.4 \
  six==1.16.0 \
  smart-open==7.1.0 \
  sniffio==1.3.1 \
  soundfile==0.12.1 \
  soxr==0.5.0.post1 \
  spacy==3.7.5 \
  spacy-legacy==3.0.12 \
  spacy-loggers==1.0.5 \
  SpeechRecognition==3.10.4 \
  srsly==2.4.8 \
  sympy==1.13.3 \
  thinc==8.2.5 \
  threadpoolctl==3.6.0 \
  tiktoken==0.7.0 \
  tokenizers==0.21.1 \
  torch==2.7.0 \
  torchaudio==2.7.0 \
  tqdm==4.66.6 \
  transformers==4.52.2 \
  triton==3.3.0 \
  typer==0.15.3 \
  typing_extensions==4.13.2 \
  urllib3==2.2.3 \
  wasabi==1.1.3 \
  weasel==0.4.1 \
  word2number==1.1 \
  wrapt==1.17.2 \
  zipp==3.20.2

echo "All packages installed successfully."
