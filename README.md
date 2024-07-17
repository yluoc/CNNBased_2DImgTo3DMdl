#### CNNBased_2DImgTo3DMdl

Predicting 3D model from corresponding 2D image (such as square to cube, circle to sphere, triangle to pyramid) using CNN.

### **Adjusting file path before running project**

To generate 2D images and 3D models dataset, using
```
python 3DShapeGenerator.py
```
To train CNN model, using
```
python train_model.py
```
If a GPU is available, using
```
python train_model_gpu.py
```
To predict the 3D model, using
```
python model_prediction.py
```
To view the .obj file generated, recommand using [https://3dviewer.net/] to read the .obj file.
