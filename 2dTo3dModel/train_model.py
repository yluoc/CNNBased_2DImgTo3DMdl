import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging verbosity

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from cnnModel import cnnModel
from dataPreprocess import dataPreprocess

class TrainModel:
    def __init__(self, img_path, mdl_path, obj_num):
        self.data_preprocess = dataPreprocess(img_path, mdl_path, obj_num)
        self.model = None

        # Check for GPU and set mixed precision policy if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("Mixed precision policy set.")
            except Exception as e:
                print(f"Error setting mixed precision policy: {e}")
        else:
            print("No compatible GPU found. Running on CPU.")

    def load_obj(self, filename):
        vertices = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(vertices)

    def prepare_data(self):
        mapped_dataset = self.data_preprocess.shape_model_match()
        
        x_train, y_train = [], []
        for match in mapped_dataset:
            img_filepath, mdl_filepath, _, _ = match

            img = self.data_preprocess.image_preprocess(img_filepath)
            model = self.load_obj(mdl_filepath)
            print(mdl_filepath, img_filepath)
            
            if img is not None and model is not None:
                x_train.append(img)
                y_train.append(model)
            else:
                print(f"Skipping pair: {img_filepath}, {mdl_filepath}")

        x_train = np.array(x_train)
        y_train = np.array(y_train, dtype=object)
        
        # Ensure all y_train entries have the same shape
        max_vertices = max(len(vertices) for vertices in y_train)
        y_train_padded = np.zeros((len(y_train), max_vertices, 3))
        for i, vertices in enumerate(y_train):
            y_train_padded[i, :len(vertices), :] = vertices

        # Flatten y_train_padded to match the model's output shape
        y_train_flattened = y_train_padded.reshape((len(y_train_padded), -1))

        return x_train, y_train_flattened

    def similarity_percentage(self, y_true, y_pred):
        criterion = MeanSquaredError()
        mse = criterion(y_true, y_pred).numpy()
        max_mse = np.mean(y_true ** 2)
        similarity = (1 - mse / max_mse) * 100
        return max(0, similarity)

    def create_dataset(self, x, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def train_model(self, epochs=10000, batch_size=256):
        x_train, y_train = self.prepare_data()

        if len(x_train) == 0 or len(y_train) == 0:
            print("No training data found!")
            return

        img_height, img_width, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
        output_vertices = y_train.shape[1] // 3

        self.model = cnnModel(img_height, img_width, channels, output_vertices).model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

        checkpoint_callback = ModelCheckpoint(filepath='./cnn_model_epoch.weights.h5', save_best_only=True, monitor='loss', mode='min', save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
        
        class SimilarityCallback(tf.keras.callbacks.Callback):
            def __init__(self, train_model_instance, x_train, y_train, batch_size=128):
                super().__init__()
                self.train_model_instance = train_model_instance
                self.x_train = x_train
                self.y_train = y_train
                self.batch_size = batch_size
            
            def on_epoch_end(self, epoch, logs=None):
                similarity = 0
                num_batches = max(1, len(self.x_train) // self.batch_size)
                for i in range(num_batches):
                    batch_x = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_y = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    y_pred = self.model.predict(batch_x)
                    similarity += self.train_model_instance.similarity_percentage(batch_y, y_pred)
                similarity /= num_batches
                print(f"Epoch {epoch+1}: Similarity = {similarity:.6f}%")

        train_dataset = self.create_dataset(x_train, y_train, batch_size)

        print("Starting training...")
        self.model.fit(train_dataset, epochs=epochs, callbacks=[SimilarityCallback(self, x_train, y_train, batch_size=batch_size), checkpoint_callback, reduce_lr])
        print("Training complete.")

        self.save_model()
    
    def save_model(self):
        if self.model:
            self.model.save('./cnn_model_final.keras', save_format='tf')  # Save the model in TensorFlow SavedModel format
            print("Model saved to './cnn_model_final.keras'")

"""
def main():
    img_path = './shapes2d'
    mdl_path = './shapes3d'
    obj_num = 100

    trainer = TrainModel(img_path, mdl_path, obj_num)
    trainer.train_model()

if __name__ == "__main__":
    main()
"""
