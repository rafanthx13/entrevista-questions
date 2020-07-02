---
layout: default
title: Deep Learning PT-BR
description: Deep Learning
---

## ConvNet




### What’s a convolutional layer? ‍⭐️

The idea of the convolutional layer is the assumption that the information needed for making a decision often is spatially close and thus, it only takes the weighted sum over nearby inputs. It also assumes that the networks’ kernels can be reused for all nodes, hence the number of weights can be drastically reduced. To counteract only one feature being learnt per layer, multiple kernels are applied to the input which creates parallel channels in the output. Consecutive layers can also be stacked to allow the network to find more high-level features.



### Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️

A fully-connected layer needs one weight per inter-layer connection, which means the number of weights which needs to be computed quickly balloons as the number of layers and nodes per layer is increased.



### What’s pooling in CNN? Why do we need it? ‍⭐️

Pooling is a technique to downsample the feature map. It allows layers which receive relatively undistorted versions of the input to learn low level features such as lines, while layers deeper in the model can learn more abstract features such as texture.



### How does max pooling work? Are there other pooling techniques? ‍⭐️

Max pooling is a technique where the maximum value of a receptive field is passed on in the next feature map. The most commonly used receptive field is 2 x 2 with a stride of 2, which means the feature map is downsampled from N x N to N/2 x N/2. Receptive fields larger than 3 x 3 are rarely employed as too much information is lost.

### Other pooling techniques include:

Average pooling, the output is the average value of the receptive field.
Min pooling, the output is the minimum value of the receptive field.
Global pooling, where the receptive field is set to be equal to the input size, this means the output is equal to a scalar and can be used to reduce the dimensionality of the feature map.

## Transfer Learning

### What is transfer learning? How does it work? ‍⭐️

Given a source domain D_S and learning task T_S, a target domain D_T and learning task T_T, transfer learning aims to help improve the learning of the target predictive function f_T in D_T using the knowledge in D_S and T_S, where D_S ≠ D_T,or T_S ≠ T_T. In other words, transfer learning enables to reuse knowledge coming from other domains or learning tasks.

In the context of CNNs, we can use networks that were pre-trained on popular datasets such as ImageNet. We then can use the weights of the layers that learn to represent features and combine them with a new set of layers that learns to map the feature representations to the given classes. Two popular strategies are either to freeze the layers that learn the feature representations completely, or to give them a smaller learning rate.