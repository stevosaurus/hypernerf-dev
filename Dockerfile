FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y python3-opencv colmap ffmpeg libopenexr-dev

# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 dev_user 

# RUN echo 'dev_user:dev_pw' | chpasswd

# RUN service ssh start

WORKDIR /app
COPY hypernerf_requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html

# EXPOSE 22
EXPOSE 8888