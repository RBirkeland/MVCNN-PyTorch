# MVCNN-PyTorch
## Multi-View CNN built on ResNet to classify 3D objects
A PyTorch implementation of MVCNN using ResNet, inspired by the paper by [Hang Su](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf).
MVCNN uses multiple 2D images of 3D objects to classify them. You can use the provided dataset or create your own.

![MVCNN](https://preview.ibb.co/eKcJHy/687474703a2f2f7669732d7777772e63732e756d6173732e6564752f6d76636e6e2f696d616765732f6d76636e6e2e706e67.png)

### Dependencies
* torch
* torchvision
* numpy
* tensorflow (for logging)

### Dataset
ModelNet40 12-view PNG dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view).

You can also create your own 2D dataset from 3D objects (.obj, .stl, and .off), using [BlenderPhong](https://github.com/WeiTang114/BlenderPhong)

### Setup
```bash
mkdir checkpoint
mkdir logs
```

### Train
To start training, simply point to the path of the downloaded dataset. All the other settings are optional.

```
python controller.py <path to dataset>  [--resnet N] [--epochs N] [-b N] [--lr LR]
                                        [--momentum M] [--lr-decay-freq W] [--lr-decay W]
                                        [--print-freq N] [-r PATH]
```

To resume from a checkpoint, use the -r tag together with the path to the checkpoint file.

### Tensorboard
To view training logs
```
tensorboard --logdir='logs' --port=6006
```

### TODO
Add support for pretrained models
