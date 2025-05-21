import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Paths
train_csv = "train.csv"
test_csv = "test.csv"
image_folder = "images"

# Load train CSV
train_df = pd.read_csv(train_csv)

# If your dataset has one-hot columns for diseases, convert them to a label column:
disease_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
if all(col in train_df.columns for col in disease_cols):
    train_df['label'] = train_df[disease_cols].idxmax(axis=1)

print("Train samples:", len(train_df))
print("Target labels:", train_df['label'].unique())

# Encode the target labels
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])

# Feature extraction function
def extract_features(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at path {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image at path {image_path}")
        return None
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False)
    return features

# Extract features for train images
X = []
y = []

print("Extracting features from training images...")
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    image_id = row['image_id']
    label = row['label_enc']
    img_path = os.path.join(image_folder, f"{image_id}.jpg")
    features = extract_features(img_path)
    if features is not None:
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)
print("Feature matrix shape:", X.shape)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on validation set
y_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=le.classes_))

# Load test CSV
test_df = pd.read_csv(test_csv)

# Extract features from test images
test_features = []
valid_image_ids = []

print("Extracting features from test images...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    image_id = row['image_id']
    img_path = os.path.join(image_folder, f"{image_id}.jpg")
    features = extract_features(img_path)
    if features is not None:
        test_features.append(features)
        valid_image_ids.append(image_id)

X_test = np.array(test_features)
X_test_pca = pca.transform(X_test)

# Predict test labels
test_preds = clf.predict(X_test_pca)
test_labels = le.inverse_transform(test_preds)

# Prepare submission file
submission = pd.DataFrame({
    "image_id": valid_image_ids,
    "label": test_labels
})

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")



