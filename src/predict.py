import VeggieVisionTF as vv
import preprocessing as pp
from PIL import Image
import os

def predict_image(prediction_image_path, model_path):
    # Load model
    veggie_vision = vv.VeggieVision()
    veggie_vision.load_model(model_path)

    # Load image 
    with Image.open(prediction_image_path) as img:
        # Preprocess Image and store the array
        img_processed = pp.image_preprocessing_pipeline(img)

    return veggie_vision.predict(img_processed)[0][0]


if __name__ == "__main__":
    # Load and process image we want to do prediciton on
    prediction_image_path = "data/prediction_data/prediction_image.jpg"
    model_path = "models/veggie_vision_tensorflow.h5"

    prediction = predict_image(prediction_image_path, model_path)

    print(prediction)