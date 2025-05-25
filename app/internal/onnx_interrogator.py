import asyncio

import numpy as np
import onnxruntime as rt
import random
from huggingface_hub import hf_hub_download
import os
import shutil
import io
from wand.image import Image
import csv
from ..config import logger, execution_provider, onnx_thread_quantity

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def load_labels(file) -> list[str]:
    with open(file, "r", encoding="utf-8", newline='') as csv_csv:
        reader = csv.DictReader(csv_csv)
        rows = list(reader)

    names = []
    categories = []

    for row in rows:
        names.append(row["name"])
        categories.append(int(row["category"]))

    name_series = map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x, names
    )
    tag_names = list(name_series)

    rating_indexes = list(np.where(np.array(categories) == 9)[0])
    general_indexes = list(np.where(np.array(categories) == 0)[0])
    character_indexes = list(np.where(np.array(categories) == 4)[0])

    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Interrogator:
    def __init__(self):
        self.character_indexes = None
        self.general_indexes = None
        self.rating_indexes = None
        self.tag_names = None
        self.model_target_size = None
        self.last_loaded_repo = None
        self.last_loaded_model = None
        self.model: rt.InferenceSession = None
        self.input_name = None
        self.label_name = None

    @staticmethod
    def download_model(model_repo):
        model_dir = os.path.join("models", model_repo)
        os.makedirs(model_dir, exist_ok=True)

        csv_filename = LABEL_FILENAME
        model_filename = MODEL_FILENAME
        csv_path = os.path.join(model_dir, csv_filename)
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(csv_path):
            logger.info(f"Download csv file for {model_repo}")
            csv_path_remote = hf_hub_download(repo_id=model_repo, filename=LABEL_FILENAME)
            shutil.copy(csv_path_remote, csv_path)

        if not os.path.exists(model_path):
            logger.info(f"Download model file for {model_repo}")
            model_path_remote = hf_hub_download(repo_id=model_repo, filename=MODEL_FILENAME)
            shutil.copy(model_path_remote, model_path)

        return csv_path, model_path

    def load_model(self, model_repo: str):

        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        sep_tags = load_labels(csv_path)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        providers = list({execution_provider, 'CPUExecutionProvider'})
        session_options = rt.SessionOptions()
        session_options.inter_op_num_threads = onnx_thread_quantity

        self.model = rt.InferenceSession(model_path, providers=providers, session_options=session_options)
        shape_type, height, width, tensor_size = self.model.get_inputs()[0].shape

        self.model_target_info = {
            "current_model": model_path,
            "shape_type": shape_type,
            "height": height,
            "width": width,
            "tensor_size": tensor_size
        }
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name
        logger.info(f"Loaded model: {model_repo}")
        for inp in self.model.get_inputs():
            logger.info(inp)

    def predict(self, image: np.ndarray, general_thresh, character_thresh):

        predictions = self.model.run([self.label_name], {self.input_name: image})[0]

        labels = list(zip(self.tag_names, predictions[0].astype(float)))
        # TODO fix i iterator pls
        ratings = [labels[i] for i in self.rating_indexes]
        ratings.sort(key=lambda x: x[1], reverse=True)

        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > general_thresh]

        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > character_thresh]

        return ratings, general_res, character_res

    async def async_predict(self, image: np.ndarray, general_thresh: float, character_thresh: float):

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def ort_callback(ort_outputs, user_data, err_msg_str):
            if err_msg_str:
                loop.call_soon_threadsafe(
                    future.set_exception,
                    RuntimeError(f"ONNX Runtime async inference failed: {err_msg_str}")
                )
                return
            try:
                predictions_tensor = ort_outputs[0]

                labels = list(zip(self.tag_names, predictions_tensor[0].astype(float)))

                # Defensive checks for index bounds
                ratings = [labels[i] for i in self.rating_indexes if i < len(labels)]
                ratings.sort(key=lambda x: x[1], reverse=True)

                general_names = [labels[i] for i in self.general_indexes if i < len(labels)]
                general_res = [x for x in general_names if x[1] > general_thresh]

                character_names = [labels[i] for i in self.character_indexes if i < len(labels)]
                character_res = [x for x in character_names if x[1] > character_thresh]

                result_tuple = (ratings, general_res, character_res)

                loop.call_soon_threadsafe(future.set_result, result_tuple)

            except Exception as e:
                # Catch any exceptions during result processing in the callback
                loop.call_soon_threadsafe(future.set_exception, e)

        input_feed = {self.input_name: image}
        output_names = [self.label_name]

        # Execute the model asynchronously.
        # The callback (ort_callback) will be invoked when inference is complete or an error occurs.
        # We pass None for user_data as the callback uses closure to access necessary variables.
        self.model.run_async(output_names, input_feed, ort_callback, None)

        # Await the future; this will pause async_predict until the future is resolved (result or exception).
        return await future
