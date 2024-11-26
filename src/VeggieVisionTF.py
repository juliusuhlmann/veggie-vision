import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os


class VeggieVision:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.build_model()

    def build_model(self):
        # Use MobileNetV2 as pretrained base model
        self.base_model = MobileNet(input_shape=self.input_shape, include_top=False, weights='imagenet')
        self.base_model.trainable = False # Pretrained base model should not be trained during training

        self.model = models.Sequential([
        self.base_model,
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')
    ])


    # Defining Operations for data augmentation
    def augment_data(self, X_train):
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range= 360,
            horizontal_flip=True, 
            zoom_range=0.1, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            brightness_range=[0.8, 1.2], 
            shear_range=0.1, 
            fill_mode='nearest'
        )
        self.datagen.fit(X_train)

    def fit(self, X, y, batch_size=32, epochs=20, learning_rate=1e-4, validation_split=0.2):
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        self.augment_data(X_train)

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), 
                           loss='mean_squared_error', metrics=['mae'])

        history = self.model.fit(self.datagen.flow(X_train, y_train, batch_size=batch_size),
                                 validation_data=(X_val, y_val), epochs=epochs)
        print(f'Validation MAE after initial training: {self.model.evaluate(X_val, y_val)[1]}')

        return history

    # Function to sequentially unfreeze base model layers
    def fine_tune(self, X, y, batch_size=16, epochs=10, fine_tune_learning_rate=1e-5, n_unfreeze_layers=20, validation_split=0.2):
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

        # Unfreeze last n layers
        for layer in self.base_model.layers[-n_unfreeze_layers:]:
            layer.trainable = True

        # Make sure layers before that remain frozen
        for layer in self.base_model.layers[:-n_unfreeze_layers]:
            layer.trainable = False

        self.model.compile(optimizer=Adam(learning_rate=fine_tune_learning_rate), 
                           loss='mean_squared_error', metrics=['mae'])

        history = self.model.fit(self.datagen.flow(X_train, y_train, batch_size=batch_size),
                                 validation_data=(X_val, y_val), epochs=epochs)
        print(f'Validation MAE after fine-tuning: {self.model.evaluate(X_val, y_val)[1]}')

        return history

    # Evaluate model on the test dataset
    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f'MSE: {loss}, MAE: {mae}')

    # Make predictions on any dataset X
    def predict(self, X):
        return self.model.predict(X)
    
    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")



if __name__ == "__main__":
     # Load the data
    X = np.load("data/processed_data/X.npy")
    y = np.load("data/processed_data/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Model
    veggie_vision = VeggieVision()

    # Train last two layers
    veggie_vision.fit(X_train, y_train, learning_rate=1e-4, epochs = 15)
    veggie_vision.fit(X_train, y_train, learning_rate=1e-5, epochs = 10)

    # Fine-tune last layer of base model
    veggie_vision.fine_tune(X_train, y_train, fine_tune_learning_rate = 1e-6, n_unfreeze_layers = 1, epochs = 5)

    # Evaluate Model
    veggie_vision.evaluate(X_test, y_test)

    # Save model
    veggie_vision.model.save('models/veggie_vision_tf.h5')