import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from cnnModel import cnnModel
from dataPreprocess import dataPreprocess
from PIL import Image
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure TensorFlow uses the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available")
        # Enable mixed precision training only if GPU is available
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    except:
        print("Invalid device or cannot modify virtual devices once initialized.")
else:
    print("No GPU available, running on CPU.")

class TrainModel:
    def __init__(self, img_path, mdl_path, obj_num, img_height, img_width, channels, output_vertices):
        self.data_preprocess = dataPreprocess(img_path, mdl_path, obj_num)
        self.model = None
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.output_vertices = output_vertices

    def prepare_data(self):
        mapped_dataset = self.data_preprocess.shape_model_match()
        
        x_train, y_train = [], []
        for match in mapped_dataset:
            img_filepath, mdl_filepath, _, _, _ = match

            img = self.image_preprocess(img_filepath)
            model = self.data_preprocess.model_preprocess(mdl_filepath)

            if img is not None and model is not None:
                x_train.append(img)
                y_train.append(model)
            else:
                print(f"Skipping pair: {img_filepath}, {mdl_filepath}")

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Print the shape of y_train before reshaping
        print(f"Shape of y_train before reshaping: {y_train.shape}")

        # Calculate the correct number of output vertices
        vertices = y_train.shape[2]  # Assuming the third dimension of y_train is the number of vertices
        self.output_vertices = vertices

        # Remove unnecessary dimensions and reshape y_train
        y_train = np.squeeze(y_train, axis=1)  # Remove the single-dimensional entries
        y_train = y_train.reshape((y_train.shape[0], vertices, 3))  # Ensure correct shape

        # Print the shape of y_train after reshaping
        print(f"Shape of y_train after reshaping: {y_train.shape}")

        return x_train, y_train

    def image_preprocess(self, img_filepath):
        """
        Preprocesses the image to the required format.
        
        :param img_filepath: Path to the image file.
        :return: Preprocessed image as a NumPy array.
        """
        image = Image.open(img_filepath).convert("RGB")  # Ensure the image is converted to RGB
        image = image.resize((self.img_width, self.img_height))
        image = np.array(image)
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def calculate_reward(self, outputs, target):
        mse = tf.reduce_mean(tf.square(outputs - target))
        reward = 1 / (mse + 1e-7)  # Adding a small epsilon to avoid division by zero
        return reward

    def custom_loss(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        reward = self.calculate_reward(y_pred, y_true)
        return mse - reward

    def load_latest_checkpoint(self):
        # Find the latest checkpoint file
        checkpoints = [f for f in os.listdir() if f.startswith('cnn_model_epoch_') and f.endswith('.h5')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        return latest_checkpoint

    def lr_schedule(self, epoch, lr):
        if epoch > 10:
            lr = lr * 0.1
        return lr

    def train_model(self, epochs=500, batch_size=32):
        x_train, y_train = self.prepare_data()

        # Data augmentation
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
        train_dataset = datagen.flow(x_train, y_train, batch_size=batch_size)

        self.model = cnnModel(self.img_height, self.img_width, self.channels, self.output_vertices).model

        latest_checkpoint = self.load_latest_checkpoint()
        if latest_checkpoint:
            try:
                print(f"Resuming from checkpoint: {latest_checkpoint}")
                self.model.load_weights(latest_checkpoint)
            except ValueError as e:
                print(f"Error loading weights: {e}")
                print("Starting training from scratch.")

        self.model.compile(optimizer='adam', loss=self.custom_loss)
        self.model.summary()

        checkpoint = ModelCheckpoint(filepath='./models/cnn_model_epoch_{epoch:02d}.h5', 
                                     save_freq='epoch', 
                                     save_best_only=False, 
                                     verbose=1)

        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
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
                    similarity += self.train_model_instance.calculate_reward(y_pred, batch_y)
                similarity /= num_batches
                print(f"Epoch {epoch+1}: Similarity = {similarity:.6f}%")

        initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) if latest_checkpoint else 0

        print("Starting training...")
        self.model.fit(train_dataset, epochs=1000, initial_epoch=initial_epoch, callbacks=[SimilarityCallback(self, x_train, y_train, batch_size=batch_size), checkpoint, early_stopping, lr_scheduler])
        print("Training complete.")

        self.save_model()
    
    def save_model(self):
        if self.model:
            self.model.save('./cnn_model_final.keras')
            print("Model saved to './cnn_model_final.keras'")

def main():
    img_path = './shapes2d'
    mdl_path = './shapes3d'
    obj_num = 1

    img_height = 64  # Set the appropriate image height
    img_width = 64   # Set the appropriate image width
    channels = 3     # Set the number of channels (e.g., 3 for RGB images)
    output_vertices = 1024  # Set the number of output vertices based on the actual data

    trainer = TrainModel(img_path, mdl_path, obj_num, img_height, img_width, channels, output_vertices)
    trainer.train_model()

if __name__ == "__main__":
    main()
