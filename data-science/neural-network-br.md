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


