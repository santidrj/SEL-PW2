import os

import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = "data"


def load_iris():
    df = pd.read_csv(os.path.join(DATA_DIR, "iris.csv"))
    return df.drop(columns="class"), df["class"]


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
