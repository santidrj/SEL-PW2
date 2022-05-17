import os

import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = "data"


def load_iris():
    return load_dataset(os.path.join(DATA_DIR, "iris.csv"))


def load_heart():
    df = pd.read_csv(
        os.path.join(DATA_DIR, "heart.csv"),
        dtype={
            "Age": "category",
            "Sex": "category",
            "ChestPainType": "category",
            "RestingBP": "category",
            "Cholesterol": "category",
            "FastingBS": "category",
            "RestingECG": "category",
            "MaxHR": "category",
            "ExerciseAngina": "category",
            "Oldpeak": np.float,
            "ST_Slope": "category",
            "HeartDisease": np.int,
        },
    )
    df["HeartDisease"] = df["HeartDisease"].map({0: "No", 1: "Yes"})
    return df.drop(columns="HeartDisease"), df["HeartDisease"].astype("category")


def load_nursery():
    df = pd.read_csv(os.path.join(DATA_DIR, "nursery.csv"), dtype="category")
    df.rename(columns={"final evaluation": "class"}, inplace=True)
    return df.drop(columns="class"), df["class"]


def load_dataset(dataset_path: str, has_class=True):
    """
    Load a CSV dataset.

    :param dataset_path: The path to the dataset.
    :param has_class: If the dataset contains the class labels. If True it assumes that are at the last column.
    :return: A tuple with the data and the class labels.
    """

    df = pd.read_csv(dataset_path)
    if has_class:
        return df.drop(columns=df.columns[-1]), df[df.columns[-1]].astype("category")
    return df
