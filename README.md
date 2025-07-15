# Audio-processor
An audio processing module:

cd C:\Projects\Audio-processor

docker build . -t audio-processor

docker run -it --rm -v "%cd%\input:/app/input" -v "%cd%\segments:/app/segments" audio-processor
