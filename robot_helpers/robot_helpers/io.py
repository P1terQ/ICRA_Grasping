import json
from pathlib import Path
import pickle
import yaml


def load_json(path):
    with Path(path).open("r") as f:
        cfg = json.load(f)
    return cfg


def load_pickle(path):
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data


def load_yaml(path):
    with Path(path).open("r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_pickle(data, path):
    with Path(path).open("wb") as f:
        pickle.dump(data, f)
