import logging
import json
import os
import time

import openai
import sentry_sdk
from flask import Flask, request, jsonify
from sentry_sdk.integrations.flask import FlaskIntegration

from pyChatGPT import ChatGPT


sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CONFIG_NAME = os.environ.get("CONFIG_NAME")
DEFAULT_CONFIDENCE = 0.9
ZERO_CONFIDENCE = 0.0

with open(CONFIG_NAME, "r") as f:
    chatgpt_init_params = json.load(f)
logging.info(f"ChatGPT initialization parameters: {chatgpt_init_params}")

# see pyChatGPT readme for info https://github.com/terry3041/pyChatGPT/blob/main/README.md
chatgpt_api = ChatGPT(**chatgpt_init_params)
logging.info("ChatGPT successfully initialized")


def generate_responses(instruction, context, continue_last_uttr=False):
    if continue_last_uttr:
        dialog_context = instruction + "\n" + "\n".join(context)
    else:
        dialog_context = instruction + "\n" + "\n".join(context) + "\n" + "AI:"
    logger.info(f"context inside generate_responses seen as: {[dialog_context]}")
    response = chatgpt_api.send_message(context).get('message', "")
    return [response]


try:
    example_response = generate_responses(
        "",
        ["Question: What is the goal of SpaceX? Answer: To revolutionize space transportation. "],
        continue_last_uttr=False,
    )
    logger.info(f"example response: {example_response}")
    logger.info("openai-api-lm is ready")
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e


@app.route("/respond", methods=["POST"])
def respond():
    st_time = time.time()
    contexts = request.json.get("dialog_contexts", [])
    try:
        responses = []
        confidences = []
        for context in contexts:
            outputs = generate_responses("", context)
            logger.info(f"openai-api-lm result: {outputs}")
            for response in outputs:
                if len(response) >= 3:
                    # drop too short responses
                    responses += [response]
                    confidences += [DEFAULT_CONFIDENCE]
                else:
                    responses += [""]
                    confidences += [ZERO_CONFIDENCE]

    except Exception as exc:
        logger.exception(exc)
        sentry_sdk.capture_exception(exc)
        responses = [[""]] * len(contexts)
        confidences = [[ZERO_CONFIDENCE]] * len(contexts)

    total_time = time.time() - st_time
    logger.info(f"transformers_lm exec time: {total_time:.3f}s")
    return jsonify(list(zip(responses, confidences)))
