import numpy as np
import onnxruntime as rt
import pandas as pd
import random
from huggingface_hub import hf_hub_download
import os
import shutil
import io
from wand.image import Image

# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

MODEL_MAPPING = {
    "SWINV2_MODEL_DSV3_REPO": SWINV2_MODEL_DSV3_REPO,
    "CONV_MODEL_DSV3_REPO": CONV_MODEL_DSV3_REPO,
    "VIT_MODEL_DSV3_REPO": VIT_MODEL_DSV3_REPO,
    "VIT_LARGE_MODEL_DSV3_REPO": VIT_LARGE_MODEL_DSV3_REPO,
    "EVA02_LARGE_MODEL_DSV3_REPO": EVA02_LARGE_MODEL_DSV3_REPO
}

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
        self.model = None
        self.input_name = None
        self.label_name = None

    def download_model(self, model_repo):
        model_name = next(key for key, value in MODEL_MAPPING.items() if value == model_repo)
        model_dir = os.path.join("models", model_name)
        os.makedirs(model_dir, exist_ok=True)

        csv_filename = LABEL_FILENAME
        model_filename = MODEL_FILENAME
        csv_path = os.path.join(model_dir, csv_filename)
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(csv_path):
            csv_path_remote = hf_hub_download(repo_id=model_repo, filename=LABEL_FILENAME)
            shutil.copy(csv_path_remote, csv_path)

        if not os.path.exists(model_path):
            model_path_remote = hf_hub_download(repo_id=model_repo, filename=MODEL_FILENAME)
            shutil.copy(model_path_remote, model_path)

        return csv_path, model_path

    def load_model(self, model_name):
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Unknown model: {model_name}")

        model_repo = MODEL_MAPPING[model_name]

        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)
        # TODO use https://docs.python.org/3/library/csv.html
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
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name

    def prepare_image(self, image_input):
        if isinstance(image_input, io.BytesIO):
            if image_input.getbuffer().nbytes == 0:
                raise ValueError("Empty BytesIO object provided")
            image_input.seek(0)
            try:
                with Image(blob=image_input.getvalue()) as img:
                    image = img.clone()
            except Exception as e:
                raise ValueError(f"Error reading image from BytesIO: {str(e)}")
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ValueError(f"Image file not found: {image_input}")
            with Image(filename=image_input) as img:
                image = img.clone()
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                raise ValueError("NumPy array must be of type uint8")
            if image_input.ndim == 2:  # Grayscale
                with Image.from_array(image_input) as img:
                    image = img.clone()
                    image.type = 'grayscale'
            elif image_input.ndim == 3 and image_input.shape[2] in [3, 4]:  # RGB or RGBA
                with Image.from_array(image_input) as img:
                    image = img.clone()
            else:
                raise ValueError("Unsupported NumPy array shape")
        else:
            raise ValueError("Unsupported image input type")

        target_size = self.model_target_size

        # Ensure the image has an alpha channel
        if image.alpha_channel:
            image.alpha_channel = 'remove'

        # Create a white background and composite the image over it
        with Image(width=image.width, height=image.height, background='white') as background:
            background.composite(image, 0, 0)
            image = background.clone()

        # Make the image square by padding
        max_dim = max(image.width, image.height)
        pad_left = (max_dim - image.width) // 2
        pad_top = (max_dim - image.height) // 2

        with Image(width=max_dim, height=max_dim, background='white') as padded_image:
            padded_image.composite(image, left=pad_left, top=pad_top)
            image = padded_image.clone()

        # Resize if necessary
        if max_dim != target_size:
            image.resize(target_size, target_size, filter='cubic')

        # Convert to numpy array
        image_array = np.array(image)

        # Ensure the image is in RGB format
        if image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]

        # Convert RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0).astype(np.float32)

    def _predict(self, image_input: Image, general_thresh, character_thresh):
        if isinstance(image_input, np.ndarray) and image_input.ndim == 4:
            image = image_input
        else:
            image = self.prepare_image(image_input)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings = [labels[i] for i in self.rating_indexes]
        ratings.sort(key=lambda x: x[1], reverse=True)

        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > general_thresh]

        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > character_thresh]

        return ratings, general_res, character_res

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
