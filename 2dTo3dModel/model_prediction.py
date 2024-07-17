import trimesh
import numpy as np
import tensorflow as tf
from dataPreprocess import dataPreprocess
from scipy.spatial import ConvexHull

class pred_model:
    def __init__(self, input_img_path, output_mdl_path, model_path):
        self.input_img_path = input_img_path
        self.output_mdl_path = output_mdl_path
        self.model_path = model_path
        self.data_preprocess = dataPreprocess('./shapes2d', './shapes3d', 1)

    def predict(self):
        cnnModel = tf.keras.models.load_model(self.model_path)

        input_img = self.data_preprocess.image_preprocess(self.input_img_path)
        if input_img is None:
            print(f"Error: Failed to preprocess the input image {self.input_img_path}")
            return

        input_img = np.expand_dims(input_img, axis=0)

        predicted_vertices = cnnModel.predict(input_img)
        predicted_vertices = np.reshape(predicted_vertices, (-1, 3))

        predicted_mesh = self.create_mesh(predicted_vertices)

        if predicted_mesh:
            trimesh.exchange.export.export_mesh(predicted_mesh, self.output_mdl_path)
            print(f"Model saved to {self.output_mdl_path}")

    def create_mesh(self, vertices):
        """
        Create a mesh from vertices using Convex Hull.
        """
        if len(vertices) < 4:
            print("Error: Not enough vertices to form the mesh.")
            return None

        try:
            # Perform Convex Hull triangulation
            hull = ConvexHull(vertices)
            faces = hull.simplices

            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            print(f"Error creating mesh: {e}")
            return None
"""
# For testing purpose
def main():
    input_img_path = './shapes2d/square_0.png'
    output_mdl_path = './a.obj'
    model_path = './cnn_model_final.keras'  # Ensure this matches the saved model format

    predictor = pred_model(input_img_path, output_mdl_path, model_path)
    predictor.predict()

if __name__ == "__main__":
    main()
"""
