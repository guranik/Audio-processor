#!/bin/bash

sudo apt-get update && sudo apt-get install -y \
    wget \
    ffmpeg \
    libsndfile1 \
    git \
    python3-venv

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

mkdir -p whisper_models

wget -q https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt \
    -O whisper_models/small.pt

mkdir -p ~/.cache/torch/hub

git clone https://github.com/snakers4/silero-vad ~/.cache/torch/hub/snakers4_silero-vad_master

mkdir -p input output

cp input.wav input/
