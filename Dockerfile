FROM ubuntu:latest
RUN apt-get update && apt-get install ffmpeg -y
RUN apt install python3-venv python3-pip -y
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY src /src
CMD ["python3", "src/main.py"]