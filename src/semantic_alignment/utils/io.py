from pathlib import Path
import torch
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_pt(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
