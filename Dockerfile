FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    libsndfile1 \
	git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY audio_processor.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/whisper-models \
    && wget https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt \
    -O /app/whisper-models/small.pt

RUN mkdir -p /root/.cache/torch/hub \
    && git clone https://github.com/snakers4/silero-vad /root/.cache/torch/hub/snakers4_silero-vad_master

VOLUME /app/input
VOLUME /app/segments


ENTRYPOINT ["python", "audio_processor.py"]
