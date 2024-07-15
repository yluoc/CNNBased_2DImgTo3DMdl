import os
import cv2
import trimesh
import numpy as np
import pandas as pd
import importlib.util
from PIL import Image

# function to import model from generator files
def import_model_from_outside(model_name, filepath):
    spec = importlib.util.spec_from_file_location(model_name, filepath)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model

file_path1 = os.path.join(os.path.dirname(__file__), '2DShapeGenerator.py')
file_path2 = os.path.join(os.path.dirname(__file__), '3DShapeGenerator.py')

img_generator_model = import_model_from_outside('2DShapeGenerator', file_path1)
mdl_generator_model = import_model_from_outside('3DShapeGenerator', file_path2)

class dataPreprocess():
    def __init__(self, image_path, model_path, obj_num=0):
        self.image_path = image_path
        self.model_path = model_path
        self.obj_num = obj_num

    # function to match related shape to model
    def shape_model_match(self):
        color_RGB = img_generator_model.random_color_tuple()
        img_generator_model.image_generator(self.obj_num, color_RGB)
        mdl_generator_model.model_generator(self.obj_num, color_RGB)
        shape_model_map = {'triangle': 'pyramid', 'square': 'cube', 'circle': 'sphere'}
        shape_info = pd.read_csv(self.image_path+'/shape_colors.csv')
        model_info = pd.read_csv(self.model_path+'/shape_colors.csv')

        shape_list = shape_info.values.tolist()
        model_list = model_info.values.tolist()

        matching_result = []  # shape_path, model_path, model, shape, color
        for shape in shape_list:
            for model in model_list:
                if shape[2] == model[2] and shape_model_map.get(shape[1]) == model[1]:
                    matching_result.append([shape[0], model[0], shape[1], model[1], shape[2]])

        return matching_result
    
    # function to preprocess image(resize, normalize)
    def image_preprocess(self, img_filedir):
        imgs = []
        for filename in os.listdir(img_filedir):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(img_filedir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                img = img = img/255.0
                imgs.append(img)
        return np.array(imgs)
    
    # function to rotate img
    def rotate_img(self, img, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    # function to flip img
    def filp_img(self, img, flip_code):
        return cv2.flip(img, flip_code)

    # function to scale img
    def scale_img(self, img, scale_factor):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)))

    # function to augment imgs
    def augment_imgs(self, imgs):
        augmented_imgs = []
        for img in imgs:
            augmented_imgs.append(img)

            for angle in [15, -15]:
                augmented_imgs.append(self.rotate_img(img, angle))

            for flip_code in [0, 1]:
                augmented_imgs.append(self.filp_img(img, flip_code))
            
            for scale_factor in [0.8, 1.2]:
                augmented_imgs.append(self.scale_img(img, scale_factor))
        
        return np.aray(augmented_imgs)

    # function to preprocess model
    def model_preprocess(self, mdl_filedir):
        models = []
        for filename in os.listdir(mdl_filedir):
            if filename.endswith('.obj'):
                mesh = trimesh.load(os.path.join(mdl_filedir, filename))
                
                # normalize the model
                mesh.apply_translation(-mesh.centroid)
                mesh.apply_scale(1/mesh.scale)
                
                # convert to voxel grid
                voxel_grid = mesh.voxelized(pitch=0.1).filled()

                voxel_grid = np.pad(voxel_grid, ((0, 32 - voxel_grid.shape[0]),
                                                 (0, 32 - voxel_grid.shape[1]),
                                                 (0, 32 - voxel_grid.shape[2])), 'constant')
                
                models.append(voxel_grid)
            
            return np.array(models)
