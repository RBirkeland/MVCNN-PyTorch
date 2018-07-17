# MVCNN-PyTorch
## Multi-View CNN built on ResNet to classify 3D objects
A PyTorch implementation of MVCNN using ResNet, inspired by the paper by [Hang Su](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf).

![MVCNN](https://preview.ibb.co/eKcJHy/687474703a2f2f7669732d7777772e63732e756d6173732e6564752f6d76636e6e2f696d616765732f6d76636e6e2e706e67.png)

### Dependencies
* torch
* torchvision
* numpy
* tensorflow (for logging)

### Dataset
ModelNet40 12-view PNG dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view).

Extract the dataset in the root folder.

### Setup
```bash
mkdir checkpoint
mkdir logs
```

### Train
```
python controller.py
```
To resume training from checkpoint, change 'resume = True' in controller.py

### Tensorboard
To view training logs
```
tensorboard --logdir='logs' --port=6006
```
