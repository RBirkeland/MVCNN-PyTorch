# MVCNN-PyTorch
## Multi-View CNN built on ResNet to classify 3D objects
A PyTorch implementation of MVCNN using ResNet, inspired by the paper by [Hang Su](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf).

Reaches validation accuracy of 87% after 20 epochs using ResNet18

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
