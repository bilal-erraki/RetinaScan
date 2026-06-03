import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_images_path = os.path.join(DATA_DIR, "train_images")

df = pd.read_csv(train_csv_path)

print("Dataset loaded successfully")
print(df.head())
print("\nClass distribution:")
print(df["diagnosis"].value_counts().sort_index())

# Add image paths
df["image_path"] = df["id_code"].apply(
    lambda x: os.path.join(train_images_path, x + ".png")
)

# Check if all images exist
df["exists"] = df["image_path"].apply(os.path.exists)
print("\nMissing images:", len(df[df["exists"] == False]))

# Plot class distribution
plt.figure(figsize=(8, 5))
df["diagnosis"].value_counts().sort_index().plot(kind="bar")
plt.title("Class Distribution - APTOS 2019")
plt.xlabel("Diagnosis")
plt.ylabel("Number of images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Show sample images
plt.figure(figsize=(12, 8))

for i in range(9):
    row = df.iloc[i]
    img = Image.open(row["image_path"])

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Diagnosis: {row['diagnosis']}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Train / validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)

print("\nTrain size:", len(train_df))
print("Validation size:", len(val_df))

print("\nTrain distribution:")
print(train_df["diagnosis"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["diagnosis"].value_counts().sort_index())

# Save split files
train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val_split.csv"), index=False)

print("\nSaved:")
print("train_split.csv")
print("val_split.csv")