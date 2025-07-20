# Audio-processor
# Instruction for Docker run:
cd C:\Projects\Audio-processor

docker build . -t audio-processor

docker run -it --rm -v "%cd%\input:/app/input" -v "%cd%\segments:/app/segments" audio-processor

# Instruction for wsl run (with venv):

In CMD/Powershell:
wsl --install

In wsl:
