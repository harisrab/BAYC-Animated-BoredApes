# set base image (host OS)
FROM python:3.6.13

WORKDIR ~/makeittalk_docker_build

COPY /home/haris/BAYC-Animated-BoredApes/requirements.txt ~/makeittalk_docker_build

RUN pip install -r requirements.txt

COPY . ~/makeittalk_docker_build

CMD ["python", "./main.py"]


