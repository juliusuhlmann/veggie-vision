import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os



# Loads images(preprocessed) and labels into numpy arrays X and y
def load_data(directory_images, filepath_labels):
    """Takes image directory and label filepath, processes the pictures 
    and loads preprocessed images and labels into X and y"""

    # "labels.csv" contains image filename and label e.g. (image1.jpg, 248)
    labels = pd.read_csv(filepath_labels)
    n_images = labels.shape[0]
    
    # Initialize X and y
    X = np.zeros((n_images, 224, 224, 3), dtype=np.float32)
    y = np.zeros((n_images,1), dtype = np.int16)
    
    # Iterate through each picture in "labels.csv"
    for i in range(n_images):
        image_path = os.path.join(directory_images, labels["filename"][i])
        try:
            # Preprocess Image and store it in X and label in y
            with Image.open(image_path) as img:
                img_processed = image_preprocessing_pipeline(img)
            X[i] = img_processed[0] #img_processed is (1, 224, 224, 3)
            y[i][0] = labels["label"][i]

        except FileNotFoundError:
            print(f"File {str(labels['filename'][i])} not found.")

    return X, y


# Complete pipeline from PIL image object to numpy array
def image_preprocessing_pipeline(img):
    """Takes PIL Image object, crops its center, resizes it to 224x224, turns it to RGB, converts it to a numpy array of 
    dimensionality (1, 224, 224, 3) and uses preprocessing specific to MobileNetV2"""
    img = crop_center(img)
    img = resize(img)
    img = to_RGB(img)
    img_array = to_array(img)
    img_array = model_specific_preprocessing(img_array)
    return img_array


# Takes a PIL image object (e.g. 2000x4000 pixel)
# Returns the center square of the picture (e.g. 2000x2000 pixel)
def crop_center(img):
    # Get image dimensions
    width, height = img.size
    # Determine the size of the square (smaller of width or height)
    new_size = min(width, height)  
    # Calculate the cropping box (center square)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2    
    # Crop the image to a square
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped


# Resize PIL image object to 224x224 pixels
def resize(img, size=224):
    return img.resize((size, size))


# Converts PIL image object to RGB
def to_RGB(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


# Convert the image to a NumPy array
def to_array(img):
    # This will have shape (224, 224, 3))
    img_array = img_to_array(img)  
    # Add a batch dimension to get the shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def model_specific_preprocessing (img_array):
    return preprocess_input(img_array)



if __name__ == "__main__":
# Select directory where pictures are stored
    directory_images = "data/raw_data/raw_images/"

    # Select filepath where lables are stored
    filepath_labels = "data/raw_data/labels.csv"

    X, y = load_data(directory_images, filepath_labels)

    directory_output = "data/processed_data/"
    
    np.save(directory_output + "X.npy", X)
    np.save(directory_output + "y.npy", y)