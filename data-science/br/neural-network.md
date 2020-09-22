---
layout: default
title: Redes Neurais PT-BR
description: Questões de Redes Neurais (Conceitos básicos)
---

Step Functions
+ Sigmoid Function
  - y entre [0,1]. O intervalo é de 1
+ Hyper Bolian Tangent Function
  - y é entre [-1,1]. O intervalo é de 2
+ SoftMax function
  - Boa para distribuiçâo de probabildiade (para classificação entre mais de duas classificações)
+ Rectified Linear Unit (ReLu)
  - Uma reta positiva que vai de [0, inf]. Se valor positivo, retorna esse avlor, se neagito, retorna 0.


+ Epochs
 - Unidade de treinamento da rede. Calcular a previsâo, verificar o erros e alterar os pesos pelo BackPropagation
+ HyperParametros: Parametro como (learning-rate (lr))


### Which optimization techniques for training neural nets do you know? ‍⭐️

Gradient Descent
Stochastic Gradient Descent
Mini-Batch Gradient Descent(best among gradient descents)
Nesterov Accelerated Gradient
Momentum
Adagrad
AdaDelta
Adam(best one. less time, more efficient)

### What’s the learning rate? 👶

The learning rate is an important hyperparameter that controls how quickly the model is adapted to the problem during the training. It can be seen as the “step width” during the parameter updates, i.e. how far the weights are moved into the direction of the minimum of our optimization problem.



### What happens when the learning rate is too large? Too small? 👶

A large learning rate can accelerate the training. However, it is possible that we “shoot” too far and miss the minimum of the function that we want to optimize, which will not result in the best solution. On the other hand, training with a small learning rate takes more time but it is possible to find a more precise minimum. The downside can be that the solution is stuck in a local minimum, and the weights won’t update even if it is not the best possible global solution.

### What is Adam? What’s the main difference between Adam and SGD? ‍⭐️

Adam (Adaptive Moment Estimation) is a optimization technique for training neural networks. on an average, it is the best optimizer .It works with momentums of first and second order. The intuition behind the Adam is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity a little bit for a careful search.

Adam tends to converge faster, while SGD often converges to more optimal solutions. SGD’s high variance disadvantages gets rectified by Adam (as advantage for Adam).



### When would you use Adam and when SGD? ‍⭐️

Adam tends to converge faster, while SGD often converges to more optimal solutions.

### How do we decide when to stop training a neural net? 👶

Simply stop training when the validation error is the minimum.


## Termos

### Batch

A batch is a fixed number of examples used in one training iteration during the model training phase.

### Batch gradient descent
Batch gradient descent is an implementation of gradient descent which computes the real gradient of the loss function by taking into account all the training examples.

In practice, batch gradient descent is rarely used for deep learning applications, because calculating the real gradient from all the training examples 1) requires to store the entire training set in the processor’s cache memory (which is often not feasible) and 2) it’s slow. Instead, methods that approximate the real gradient like stochastic gradient descent or mini-batch (stochastic) gradient descent are used.

### Epoch
An epoch represents a full pass over the entire training set, meaning that the model has seen each example once. An epoch is thus the total number of examples / batch size number of training iterations.

### Gradient descent
Gradient descent is the basic algorithm used to minimize the loss based on the training set. It’s an iterative process through which the parameters of your model are adjusted and thereby gradually finding the best combination to minimize the loss. It does this by computing the gradient (or the ‘slope’) of the loss function and then ‘descending’ down it (or taking a step down the ‘slope’) towards a lower loss value.

### Hidden layer
A hidden layer is any layer in a neural network between the input layer and the output layer.

### Hyperparameter
A hyperparameter is any parameter of a model or training process that has to be set / fixed before starting the training process, i.e., they are not automatically adjusted during the training. Examples of hyperparameters include: drop rate (dropout), batch size, learning rate, number of layers, number of filters, etc.

### Loss
Loss is a measure of how well your algorithm models your dataset. It’s a numeric value which is computed by the loss function. The lower the loss, the better the performance of your model.

### Loss function
A loss function is a function that determines how errors are penalized. The goal of the loss function is to capture in a single number the total amount of errors across all training example.

### Optimizer (gradient descent)
An optimizer is a specific implementation of gradient descent which improve the performance of the basic gradient descent algorithm.

Optimizers aim to mitigate some of the challenges that are characteristic of gradient descent like: convergence to suboptimal local minima and setting the starting learning rate and it’s decay.

In practice, using a gradient descent optimizer is the go-to choice for training deep learning models.

### Output layer
The output layer is the last layer of a neural network and is the one which returns the results of your model (e.g., the class in a classification model or a value in a prediction model). On the Peltarion Platform this layer is represented by the input block.

All your models will need an output layer (i.e., output block).

### Vanishing gradient problem
The vanishing gradient problem is the phenomenon of the gradients calculated by gradient descent getting progressively smaller when moving backward in the networks from output to input layer.

This means that the weights of the nodes in early layers only change slowly (compared to later layers in a network), which means that they train and hence, learn, very slowly or not at all.

## Activision FUnctions

### O que é Activation function
An activation function is a non-linear function which takes the weighted sum of all the inputs to a node and maps them to values in the range of 0 to 1 (e.g., Sigmoid), 0 to ∞ (e.g. ReLu) or -1 to 1 (e.g., TanH).

Its non-linear nature is what allows neural networks to model any kind of function (it makes them a universal function approximator).

Use an activation function in every node of a network. On the Peltarion Platform, all nodes are automatically assigned an activation function, which can be changed as one of the configurable parameters of the blocks that have nodes.

### ReLU (rectified linear unit) activation function

ReLU (rectified linear unit) activation function
The ReLU (rectified linear unit) is a non-linear function that gives the same output as input if the input is above 0, otherwise the output will be 0.

It is cheap to compute and works well for many applications. It also helps prevent the vanishing gradients problem.

It is the go-to activation function for many neural networks.

f(x) = max(x, 0);
f(x)=max(x,0);

### Sigmoid activation function
The sigmoid activation function generates a smooth non-linear curve that maps the incoming values between 0 and 1.

The sigmoid function works well for a classifier model but it has problems with vanishing gradients for high input values, that is, y change very slow for high values of x. Unlike the softmax activation function, the sum of all the outputs doesn’t have to be 1 when sigmoid is used as an activation function in the output layer. This means that each output node with a sigmoid activation function acts independently on each input, so more than one output node can fire at the same time.

The sigmoid function is often used together with the loss function binary crossentropy.

Use for binary classification or multilabel classification problems.

f(x) = \frac{1}{1 + e^{-x}}
f(x)= 
1+e 
−x
 
1

### Softmax activation function
The softmax activation function will calculate the relative probability of each target class over all possible target classes in the dataset given the inputs it receives. In other words it normalizes the outputs so that they sum to 1, so that they can be directly treated as probabilities over the output.

This is usefull for multiclass classification models, as the target class with the highest probability is going to be the output of the model.

It is often used in the final layer in a classification model with the categorical crossentropy as loss function.

Range: 0 to 1 - Multiple classes
\sigma(x_j) = \frac{e^{x_j}}{\sum_{k=0}^{K} e^{x_k}}
σ(x 
j
​	
 )= 
∑ 
k=0
K
​	
 e 
x 
k
​	
 
 
e 
x 
j
​	
 
 
​	
### Tanh activation function
Tanh is a scaled sigmoid activation function. The gradient is stronger for tanh than sigmoid, that is, the derivatives are steeper.

Unlike the sigmoid function, the tanh function is zero-centered, which means that it dosen’t introduce a bias in the gradients making training a network easier. The downsinde is that tanh is computationally more expensive than the sigmoid function.

Which one to use of the sigmoid or tanh depends on your requirement of gradient strength. Tanh resembles a linear function more as long as the activations of the network can be kept small. This makes the tanh network easier to compute.

f(x) = \frac{2}{1 + e^{-2x}} - 1
f(x)= 
1+e 
−2x
 
2
​	
 −1
​	
