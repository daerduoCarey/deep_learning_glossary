
# Deep Learning Glossary

Simple, opinionated explanations of various things encountered in
Deep Learning / AI / ML. 

Contributions welcome - there may be errors here!

## Contests

### ILSVRC = ImageNet Large Scale Visual Recognition Competition 

The most prominent computer vision contest, using the largest data set of images (ImageNet).
The progress in the classification task has brought CNNs to dominate the field of computer vision.

| Year          | Model                     | Top-5 Error | Layers | Paper
| ------------- |:-------------------------:| -----------:|-------:|------------------------------
| 2012          | AlexNet                   |     17.0 %  |      8 | http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
| 2013          | ZFNet                     |     17.0 %  |      8 | http://arxiv.org/abs/1311.2901
| 2014          | VGG-19                    |      8.43%  |     19 | http://arxiv.org/abs/1409.1556
| 2014          | GoogLeNet / Inception     |      7.89%  |     22 | http://arxiv.org/abs/1409.4842
| 2015          | Inception v3              |      5.6%   |        | http://arxiv.org/abs/1512.00567
| 2015          | ResNet                    |      4.49%  |    152 | http://arxiv.org/abs/1512.03385


## Techniques

### Stochastic Gradient Descent (SGD)

The original and simpliest back propagation optimization algorithm. Still used everywhere!

### SGD with Momentum 

A simple and often used improvement to SGD - follow past gradients with some weight.

### Adagrad

Another optimizer

### Adam Optimizer 

[Kingma & Ba, 2015](http://arxiv.org/pdf/1412.6980v7.pdf)

### FTRL-proximal algorithm, Follow-the-regularized-leader

[Google, 2013](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

### Rectified Linear Unit (ReLU)

Rectified linear unit is a common activation function which was first proved useful in AlexNet.
Recommend over sigmoid activiation.

```
relu(x) = max(x, 0)
```

[Nair & Hinton, 2010](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

### Parametric Rectified Linear Unit (PReLU)

http://arxiv.org/pdf/1502.01852v1

### Leaky Rectified Linear Unit

Sometimes inputs to an ReLU can get pushed way negative, in which case the
neuron can get permanently turned off (the exploding/vanishing gradients problem).
The leaky ReLU is used to combat this problem by having a small slope on the
negative side.

```
def leaky_relu(x): 
  if 0 <= x:
    return x
  else:
    return 0.01 * x
```


### Batch Normialization (BN)

Normalizing the inputs to each activation function can dramatically speed up
learning.

[Ioffe & Szegedy, 2015](http://arxiv.org/abs/1502.03167)

### Dropout

Introduced in AlexNet? Randomly zero out 50% of inputs during the forward pass.
Simple regularizer.

### LSTM = Long Short Term Memory

A type of RNN that solves the exploding/vanishing gradient problem.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

[This paper](http://arxiv.org/abs/1503.04069) is a great exploration of
variations. Concludes that [vanilla LSTM](http://www.sciencedirect.com/science/article/pii/S0893608005001206) is best.

Originally invented by [Hochreiter & Schmidhuber, 1997](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

### Attention

Not well defined. The idea of processing only part of the input at each time
step. In visual attention, this is a low resolution window that jumps around
the high resolution input image. Determining where to focus attention is
controlled by a RNN. 

Typical convolutional networks have input around 300 x 300 inputs, which is
rather low resolution. Computation complexity grows at least linearly in the
number of pixel inputs.  Attention models seem like a promising method of
handling larger inputs.

[Recurrent Models of Visual Attention](http://arxiv.org/abs/1406.6247) Attention applied to MNIST.

[Multiple Object Recognition with Visual Attention](http://arxiv.org/abs/1412.7755) Essentially v2 of the above model.

[DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/abs/1502.04623) A generative attention model


## Models

### AlexNet

Winner of ILSVRC 2012. Made a huge jump in accuracy using CNN. 
Dropout, ReLUs.

http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

### VGG-16 / VGG-19 / OxfordNet
Close second place winner in ILSVRC 2014. Very simple CNN architecure using only
3x3 convolutions, max pooling, ReLUs, dropout

### Neural Random-Access Machine (NRAM)

http://arxiv.org/pdf/1511.06392v1


### Grid-LSTM

(Kalchbrenner et al., 2015)


### Neural Turing Machine (NTM)

(Graves et al., 2014)


### Deep Q Network (DQN)

(Volodymyr Mnih Koray Kavukcuoglu David Silver Alex Graves Ioannis Antonoglou
Daan Wierstra Martin Riedmiller)

### R-CNN Regions with Convolutional Neural Networks

Object detection model.

Girshick et al., 2014


## Software

### Caffe

### TensorFlow

### Theano 

### Torch

### CuDNN

### MxNet



## Data sets

### CIFAR-10

60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

https://www.cs.toronto.edu/~kriz/cifar.html

### ImageNet

### MNIST

Handwritten digits. 28x28 images. 60,000 training images and 10,000 testing images

### IAM Handwriting Database

http://www.iam.unibe.ch/fki/databases/iam-handwriting-database

Famously used in Graves's handwriting generation RNN: http://www.cs.toronto.edu/~graves/handwriting.html

http://yann.lecun.com/exdb/mnist/

### TIMIT Speech corpus 

(Garofolo et al., 1993) 
