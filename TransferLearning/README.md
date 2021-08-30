# Instructions

## Getting our Data

The data we have used was supplied by Kaggle, a great place
to find training data for our projects.

[Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)

You will want to download this dataset and place the folders:
**seq_test** and **seq_train** inside of the empty 'Class_Data'
directory.

## Setting up your Environment

If you want to run this example, it is entirely possible to simply use
the Docker container we have been using for our previous examples.

However, the model used for this transfer learning example will
take quite a long time to train on a CPU. 

(Even though I made multiple iterations of improvements training will
take MUCH longer than you expect!)

**I HIGHLY recommend you use train this model via a GPU**

## Training VGG16 via GPU using Tensorflow and Docker

*Look at those buzz words!*

This does however, present an excellent opportunity to learn how
utilize your GPU with realtive ease!

I will recommend the method I used.

**You will need to have Docker installed on your system**
**You will also need GPU that is CUDA compatible**

Here is the guide TensorFlow provides for how to run using a Docker container:
 - https://www.tensorflow.org/install/docker

## Running your Docker container

To train this model yourself:

Just run the given command from within the directory that Transfer.py is located in
from a terminal.

I will also warn you that this Docker container is quite large,
around ~7GBs, so have a cup of tea nearby so your wait is more enjoyable.

### Windows

docker run --gpus all --rm -v %cd%:/tmp -w /tmp tensorflow/tensorflow:latest-gpu python ./Transfer.py

### Linux

sudo docker run --gpus all --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:latest-gpu python ./Transfer.py

# Notes

I wanted to include a saved model for both our model we have been working with,
as well as the VGG16 model. But the file sizes were too large for GitHub and 
I am not sure of a suitable way to supply these models to everyone. 

If you would like to request these pre-trained models or if you need assistance with importing 
and getting predictions from your model from me please reachout on GitHub

