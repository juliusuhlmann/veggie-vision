import VeggieVisionTF as vv
import preprocessing as pp
from PIL import Image
import os


# Set the working directory to the current file's directory
os.chdir(os.path.dirname(__file__))

# Load model
veggie_vision = vv.VeggieVision()
veggie_vision.load_model("../models/veggie_vision_tensorflow.h5")

# Load and process image we want to do prediciton on
prediction_image = "../data/prediction_data/prediction_image.jpg"

with Image.open(prediction_image) as img:
    # Preprocess Image and store the array
    img_processed = pp.image_preprocessing_pipeline(img)

print(veggie_vision.predict(img_processed))
