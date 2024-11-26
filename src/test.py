import numpy as np
from sklearn.model_selection import train_test_split
import VeggieVisionTF as vv

model_path = "models/veggie_vision_tensorflow.h5"

# Load the data
X = np.load("data/processed_data/X.npy")
y = np.load("data/processed_data/y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Model and load model
veggie_vision = vv.VeggieVision()
veggie_vision.load_model(model_path)

# Evaluate Model
veggie_vision.evaluate(X_test, y_test)