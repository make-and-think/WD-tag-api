import numpy as np
import onnxruntime as rt
import pandas as pd
import random
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import shutil
import io

# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

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


def load_labels(dataframe) -> list[str]:
    # TODO use https://docs.python.org/3/library/csv.html
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
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
        self.model_target_size = None
        self.last_loaded_repo = None
        self.last_loaded_model = None

    def download_model(self, model_repo):
        os.makedirs("models", exist_ok=True)
        csv_filename = LABEL_FILENAME
        model_filename = MODEL_FILENAME
        csv_path = os.path.join("models", csv_filename)
        model_path = os.path.join("models", model_filename)

        if not os.path.exists(csv_path):
            csv_path_remote = hf_hub_download(repo_id=model_repo, filename=LABEL_FILENAME)
            shutil.copy(csv_path_remote, csv_path)

        if not os.path.exists(model_path):
            model_path_remote = hf_hub_download(repo_id=model_repo, filename=MODEL_FILENAME)
            shutil.copy(model_path_remote, model_path)

        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)
        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        # gpu doesnt't work :(
        available_providers = rt.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # CUDA + CPU fallback
        else:
            providers = ['CPUExecutionProvider']

        self.model = rt.InferenceSession(model_path, providers=providers)
        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo

    def prepare_image(self, image_input):
        # TODO replace Pillow with this https://docs.wand-py.org

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGBA")
        elif isinstance(image_input, io.BytesIO):
            image = Image.open(image_input).convert("RGBA")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                raise ValueError("NumPy array must be of type uint8")
            if image_input.ndim == 2:  # Grayscale
                image = Image.fromarray(image_input, mode='L').convert("RGBA")
            elif image_input.ndim == 3 and image_input.shape[2] in [3, 4]:  # RGB or RGBA
                image = Image.fromarray(image_input, mode='RGBA' if image_input.shape[2] == 4 else 'RGB')
            else:
                raise ValueError("Unsupported NumPy array shape")
        else:
            raise ValueError("Unsupported image input type")

        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # TODO remove because already square image
        image_shape = image.size
        max_dim = max(image_shape)  # 1024
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)  # TODO use wand or numpy

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(self, image_input, general_thresh, character_thresh):
        if isinstance(image_input, np.ndarray) and image_input.ndim == 4:
            image = image_input
        else:
            image = self.prepare_image(image_input)

        input_name = self.model.get_inputs()[0].name  # TODO move to class
        label_name = self.model.get_outputs()[0].name  # TODO move to class
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings = [labels[i] for i in self.rating_indexes]
        ratings.sort(key=lambda x: x[1], reverse=True)

        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > general_thresh]

        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > character_thresh]

        return ratings, general_res, character_res
