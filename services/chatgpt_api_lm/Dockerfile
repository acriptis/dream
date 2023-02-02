FROM python:3.7.4

WORKDIR /src

COPY ./services/chatgpt_api_lm/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY . /src

RUN apt-get update && apt-get -y install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker

CMD ["/bin/bash","-c","/src/startup.sh"]

CMD gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --timeout=300
