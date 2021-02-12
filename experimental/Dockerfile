# This is a Container we can use to run Experiments in Paperspace.
#   Edit, build, and publish to DockerHub.
#   `docker build . -t bcollazo/paperspace-rl` to build.
#   `docker push bcollazo/paperspace-rl` to publish.
FROM paperspace/tensorflow:2.0.0-gpu-py3-jupyter-lab

RUN pip install --upgrade pip
RUN pip install tqdm
RUN pip install pillow
RUN pip install opencv-python

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
