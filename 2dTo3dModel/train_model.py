import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
import threading
import numpy as np
from cnnModel import *
import tensorflow as tf
from pynput import keyboard
from dataPreprocess import dataPreprocess

class train_model:
    def __init__(self, img_path, mdl_path, obj_num=0):
        self.data_preprocess = dataPreprocess(img_path, mdl_path, obj_num)
        self.checkpoint_path = "training_checkpoints/cp-{epoch:04d}.weights.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model_save_path = "trained_model/model_{epoch:04d}.keras"
        self.model_save_dir = os.path.dirname(self.model_save_path)
        self.pause_flag = False
        self.pause_status = False
        self.stop_status = False
        self.model = None
    
    def keyboard_monitor(self):
        def on_press(key):
            try:
                if key.char == 's':
                    self.pause_status = not self.pause_status

                    if self.pause_status:
                        if not self.pause_flag:
                            print("\nTraining paused. Press 's' to resume.")
                            self.pause_flag = True
                        else:
                            print("\nTraining resumed")
                            self.pause_flag = False
                
                elif key.char == 'e':
                    self.stop_status = True
                    print("\nTraining stopped and model saved.")
            except AttributeError:
                pass
        
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    
    def save_model(self, epoch):
        model_save_path = self.model_save_path.format(epoch=epoch)
        model_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model.save(model_save_path)

    def prepare_data(self):
        mapped_dataset = self.data_preprocess.shape_model_match()
        
        x_train, y_train = [], []
        for match in mapped_dataset:
            img_filepath, mdl_filepath, _, _, _ = match

            img = self.data_preprocess.image_preprocess(img_filepath)
            model = self.data_preprocess.model_preprocess(mdl_filepath)

            if img is not None and model is not None:
                x_train.append(img)
                y_train.append(model)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        return x_train, y_train

    # adjust epoches, batch_size, validation_split, patience as needed
    def trainModel(self, epochs=100, batch_size=128, validation_split=0.2):
        x_train, y_train = self.prepare_data()

        if len(x_train) == 0 or len(y_train) == 0:
            print("No training data found!")
            return

        img_height, img_width, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
        output_vertices = y_train.shape[1]

        self.model = cnnModel(img_height, img_width, channels, output_vertices).model

        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.model.load_weights(latest_checkpoint)
            print(f"Loaded weights from checkpoint: {latest_checkpoint}")

        self.model.summary()

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.checkpoint_path,
            save_weights_only = True,
            save_freq = 'epoch',
            verbose = 1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        key_monitor_thread = threading.Thread(target=self.keyboard_monitor)
        key_monitor_thread.daemon = True
        key_monitor_thread.start()

        print("Starting training...")
        try:
            for epoch in range(epochs):
                if self.stop_status:
                    break

                print(f"\nEpoch {epoch + 1}/{epochs}")
                
                while self.pause_status:
                    if not self.pause_flag:
                        print("\nTraining paused. Press 's' to resume.")
                        self.pause_flag = True
                    time.sleep(1)
                
                if self.pause_status:
                    continue

                self.model.fit(
                    x_train, y_train,
                    epochs = 1,
                    batch_size = batch_size,
                    validation_split = validation_split,
                    callbacks = [checkpoint_callback, early_stopping]
                )
                self.save_model(epoch)
                 
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current state...")
            self.model.save_weights(self.checkpoint_path.format(epoch=epoch))
            print("Model state saved")

        print("Training complete.")
    
# for testing purpose
def main():
    img_path = './shapes2d'
    mdl_path = './shapes3d'
    obj_num = 10

    trainer = train_model(img_path, mdl_path, obj_num)
    trainer.trainModel()

if __name__ == "__main__":
    main()