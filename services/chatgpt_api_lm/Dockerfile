FROM python:3.7.4

WORKDIR /src

COPY ./services/chatgpt_api_lm/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

ARG CONFIG_NAME
ENV CONFIG_NAME ${CONFIG_NAME}

COPY . /src

RUN apt-get update && apt-get install snapd && apt install xvfb && apt install chromium-browser xvfb

CMD gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --timeout=300
