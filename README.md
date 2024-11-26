# VeggieVision: Weight Prediction for Frozen Vegetables

## Project Overview
VeggieVision is a machine learning-powered application that predicts the weight of frozen vegetables based on an uploaded image. Designed to estimate weights between **1g and 250g**, this project achieves high accuracy with a **mean absolute error (MAE) of less than 19g**. The project also features a user-friendly **Graphical User Interface (GUI)** for seamless interaction.

---

## Project Structure

VeggieVision/ ├── data/ │ ├── prediction_data/ # Images for prediction │ ├── processed_data/ # Preprocessed images │ └── raw_data/ # Original dataset ├── models/ │ └── veggie_vision_tensorflow.h5 # Trained TensorFlow model ├── notebooks/ │ ├── automated_labeling.ipynb # Notebook for data labeling │ └── modelling.ipynb # Model training and evaluation ├── src/ │ ├── GUI.py # GUI implementation │ ├── predict.py # Prediction functions │ ├── preprocessing.py # Data preprocessing scripts │ ├── test.py # Testing and validation scripts │ ├── VeggieVisionPT.py # PyTorch model implementation │ └── VeggieVisionTF.py # TensorFlow model implementation


---

## How It Works

### 1. Data Preprocessing
Raw images in `data/raw_data/` are processed using `preprocessing.py` to standardize dimensions and augment data if necessary.

### 2. Model Training
- **TensorFlow model:** `VeggieVisionTF.py`  
- **PyTorch model:** `VeggieVisionPT.py`  

Training workflows are detailed in `notebooks/modelling.ipynb`.

### 3. Weight Prediction
- Use `predict.py` to test the trained model on new data in `data/prediction_data/`.
- GUI: Run `GUI.py` to upload images and view predictions in real time.

### 4. Testing
Model performance is validated using `test.py` to ensure robustness.

---

## Usage

### 1. Run the GUI
To interact with the model:

python src/GUI.py

Upload an image and view the predicted weight instantly.

### 2. Predict Programmatically
Use predict.py to predict weights:

from src.predict import predict_weight
prediction = predict_weight("data/prediction_data/example.jpg", model_path="models/veggie_vision_tensorflow.h5")
print(f"Predicted Weight: {prediction}g")

## Contact
For questions or collaboration opportunities, please reach out at your.email@example.com.