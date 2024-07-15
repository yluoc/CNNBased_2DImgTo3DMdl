#### CNNBased_2DImgTo3DMdl

Predicting 3D model from corresponding 2D image (such as square to cube, circle to sphere, triangle to pyramid) using CNN.

To generate 2D images and 3D models dataset, using
```
python 2DShapeGenerator.py
python 3DShapeGenerator.py
```
To train CNN model, using
```
python train_model.py
```
_notice:_
train_model.py has a functionality of press 's' to pause and press 's' again to resume the training process, press 'e' to stop and save already trained model, so you can not touch 's' or 'e' on keyboard while training. You can modify the code in train_model.py to disactivate this functionality.
To predict the 3D model, using
```
python model_prediction.py
```
