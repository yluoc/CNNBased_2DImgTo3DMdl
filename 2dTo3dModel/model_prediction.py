import os
import trimesh
import numpy as np
import tensorflow as tf
import trimesh.exchange
from train_model import *
from cnnModel import cnnModel
import trimesh.exchange.export
from dataPreprocess import dataPreprocess

class pred_model():
    
    def __init__(self, input_img_path, checkpoint_dir):
        self.input_img_path = input_img_path
        self.output_mdl_path = "output_mesh/output.obj"
        self.output_mdl_dir = os.path.dirname(self.output_mdl_path)
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.data_preprocess = dataPreprocess('./shapes2d', './shapes3d')
        self.train_model = train_model('./shapes2d', './shapes3d')

        x, y = self.train_model.prepare_data()
        img_height, img_width, channels = x.shape[1], x.shape[2], x.shape[3]
        output_vertices = y.shape[1]

        self.model = cnnModel(img_height, img_width, channels, output_vertices).model


    def load_weights_from_h5(self, weights_file):
        if os.path.exists(weights_file):
            self.model.load_weights(weights_file)
            print(f"Loaded weights from HDF5 file: {weights_file}")
        else:
            raise ValueError(f"Weights file {weights_file} not found")

    def predict(self):
        input_img = self.data_preprocess.image_preprocess(self.input_img_path)
        input_img = np.expand_dims(input_img, axis=0)

        predicted_vertices = self.model.predict(input_img)
        predicted_vertices = np.reshape(predicted_vertices, (-1, 3))

        predicted_mesh = trimesh.Trimesh(vertices=predicted_vertices)

        trimesh.exchange.export.export_mesh(predicted_mesh, self.output_mdl_path)


# for testing purpose
def main():
    input_img_path = './shapes2d/square_0.png'
    weights_file = './training_checkpoints/cp-0001.weights.h5'

    predictor = pred_model(input_img_path, weights_file)
    predictor.load_weights_from_h5(weights_file)
    predictor.predict()

if __name__ == "__main__":
    main()
