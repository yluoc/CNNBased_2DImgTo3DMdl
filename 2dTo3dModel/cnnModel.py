import tensorflow as tf
from tensorflow.keras import layers, models

class cnnModel:
    def __init__(self, img_height, img_width, channels, output_vertices):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.output_vertices = output_vertices
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, self.channels)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.output_vertices * 3))  # Output is the flattened vertex array

        return model
