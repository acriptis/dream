import re

import requests
from deeppavlov import build_model
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from num2words import num2words

import os

app = FastAPI()
#asr = build_model("asr.json")
tts = build_model("tts.json")


@app.post("/asr")
async def infer_asr(user_id: str, file: UploadFile = File(...)):
    # Save file
    try:
        contents = file.file.read()
        with open('to_asr.wav', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

#    os.system('/home/wenet/runtime/libtorch/bin/decoder_main --chunk-size -1 --wav_path to_asr.wav --model_path asr/final.zip --unit-path asr/units.txt > transcript.txt')
    os.system('GLOG_log_dir=/app/log GLOG_v=2 /home/wenet/runtime/libtorch/bin/grpc_client_main --hostname localhost --wav_path to_asr.wav --interval 100000 --sleep 0')
    with open('log/grpc_client_main.INFO') as f:
        loglines=f.readlines()
    result_line = None
    for line in loglines:
        line_split = line.split(maxsplit=4)
        if line_split[3].startswith('grpc_client.cc') and line_split[4].startswith('1best '):
            result_line = line_split
    if result_line:
        transcript = result_line[4].strip()
        transcript = re.sub('^1best ', '', transcript)
    else:
        # when there's no 1best line, show error
        transcript = 'ERROR: no result from grpc_client'

#    transcript = asr([file.file])[0]
#    with open('transcript.txt') as f:
#        transcript = f.read().strip()
#    transcript = re.sub("^test ", "", transcript)

    print(f'transcription is "{transcript}"')
    post_response = requests.post("http://agent:4242", json={"user_id": user_id, "payload": transcript})
    response_payload = post_response.json()
#    response_payload = {}
    response = response_payload["response"]
    print(f'response is "{response}"')
    response = re.sub(r"([0-9]+)", lambda x: num2words(x.group(0)), response)
#    response = f'The transcript is: {transcript}'
    response_payload["response"] = response
    return JSONResponse(content=response_payload, headers={"transcript": transcript})


@app.post("/tts")
async def infer_tts(text: str):
    response = re.sub(r"([0-9]+)", lambda x: num2words(x.group(0)), text)
    print(f'response is "{response}"')
    audio_response = tts([response])[0]
    return StreamingResponse(audio_response, media_type="audio/x-wav")
