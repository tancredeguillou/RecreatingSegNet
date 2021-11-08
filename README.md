# Recreating SegNet : Implementation

If you're not familiar with the SegNet algorithm, you can visit the following [website](https://medium.com/@fezancs/understanding-of-semantic-segmentation-how-segnet-model-work-to-perform-semantic-segmentation-5c426112e499).

Here, our model will contain eight levels:
- Four downscaling levels containging the following layers : Convolutional, Batch Normalisation, ReLu, Pooling.
- Four upscaling levels containing the following layers : Upsampling, Convolutional, Batch Normalisation, ReLu.
We end the model by computing Softmax and a final linear layer.

## Overview
- `image_segmentation/`: includes training and validation scripts.
- `lib/`: contains core functions, data preparation, model definition, and utility functions.

## Installation
1. This project uses [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [PyTorch 1.9.1](https://docs.conda.io/en/latest/miniconda.html).
2. To install dependencies, run:
   ```
   pip install -r requirements.txt
   ```
   NOTE: TensorBorad and tensorboardX may not be compatible on some platforms. If you encounter difficulties when installing them, just remove them from `requirements.txt`. They are used solely for visualizing your results, which is optional (but helpful) for building your model.
3. Add current project directory (which we will later denote as ${ROOT}) to PYTHONPATH environment variable.
   ```
   export PYTHONPATH=${PYTHONPATH}:${PWD}
   ```

## Data Preparation for Multi-digit MNIST Dataset
1. Unzip the multi-digit-MNIST dataset (multi-digit-mnist-dataset.zip) to ${ROOT}. You should have the following directory structure after unzipping:
   ```
   ${ROOT}
    `-- data
        `-- multi-digit-mnist 
            |-- batch00001.mat
            |...
            |-- testset001.mat
            |...

   ```

## Training on Multi-digit MNIST Dataset
To train your model, run:
```
python image_segmentation/train_mnist.py
```

## Validate the model
To validate the model after training, run:
```
python image_segmentation/validate_mnist.py
```

If you installed TensorBoard and tensorboardX, then you should have TensorBoard logs saved to `out/logs`. You can monitor the logs (loss curves, validation visualization, etc.) on <http://localhost:6006> via:
```
tensorboard --logdir out/logs --port 6006
```

## References
1. The overall structure of the code (roughly) follows [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch).
2. Multi-digit MNIST dataset was created using the script from [Recurrent Pixel Embedding for Instance Grouping](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping).
