import os
import pandas as pd

DATA_DIR = "aptos2019-blindness-detection"

train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

print(train_csv.head())
print(train_csv["diagnosis"].value_counts())
print("Number of training images:", len(os.listdir(os.path.join(DATA_DIR, "train_images"))))