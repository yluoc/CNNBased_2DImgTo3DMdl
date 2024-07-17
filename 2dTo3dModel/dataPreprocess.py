import cv2
import numpy as np
import os

class dataPreprocess:
    def __init__(self, img_path, mdl_path, obj_num):
        self.img_path = img_path
        self.mdl_path = mdl_path
        self.obj_num = obj_num

    def shape_model_match(self):
        img_files = sorted(os.listdir(self.img_path))[:self.obj_num]
        mdl_files = sorted(os.listdir(self.mdl_path))[:self.obj_num]

        dataset = []
        for img_file, mdl_file in zip(img_files, mdl_files):
            dataset.append((os.path.join(self.img_path, img_file), os.path.join(self.mdl_path, mdl_file), img_file, mdl_file))
        return dataset

    def image_preprocess(self, img_filepath):
        img = cv2.imread(img_filepath)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images to a fixed size
            img = img / 255.0  # Normalize the image
        return img
