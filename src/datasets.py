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
            "Age": np.int,
            "Sex": str,
            "ChestPainType": str,
            "RestingBP": np.int,
            "Cholesterol": np.int,
            "FastingBS": str,
            "RestingECG": str,
            "MaxHR": np.int,
            "ExerciseAngina": str,
            "Oldpeak": np.float,
            "ST_Slope": str,
            "HeartDisease": np.int,
        },
    )
    df["HeartDisease"] = df["HeartDisease"].map({0: "No", 1: "Yes"})
    return df.drop(columns="HeartDisease"), df["HeartDisease"]


def load_rice():
    data, meta = arff.loadarff(os.path.join(DATA_DIR, "Rice_Cammeo_Osmancik.arff"))
    df = pd.DataFrame(data)
    class_col = "Class"
    df[class_col] = df[class_col].str.decode("utf-8")
    return df.drop(columns=class_col), df[class_col]


def load_dataset(dataset_path: str, has_class=True):
    """
    Load a CSV dataset.

    :param dataset_path: The path to the dataset.
    :param has_class: If the dataset contains the class labels. If True it assumes that are at the last column.
    :return: A tuple with the data and the class labels.
    """

    df = pd.read_csv(dataset_path)
    if has_class:
        return df.drop(columns=df.columns[-1]), df[df.columns[-1]]
    return df
