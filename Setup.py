import pandas as pd
import numpy as np
import os
import kagglehub

def dataset():
    path = kagglehub.dataset_download("ethancratchley/email-phishing-dataset")

    print("Path to dataset files:", path)
    df = pd.read_csv(os.path.join(path, "email_phishing_data.csv"))
    return df
if __name__ == "__main__":
    df = dataset()
    df.head()
    print("Dataset loaded successfully.")
    print("Number of rows:", df.shape[0])
    