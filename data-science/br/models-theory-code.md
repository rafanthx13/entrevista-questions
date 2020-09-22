---
layout: default
title: Modelos, Teoria e Código PT-BR
description: Questões de DataScience e Machine Learning
---

# Modelos, Teoria e Prática

## AdaBoost

Adaptative Boosting ou AdaBoost é um modelo adaptativo em conjunto (ensemble).


O nome "AdaBoost" deriva de Adaptive Boosting (em português, impulso ou estímulo adaptativo). O AdaBoost é adaptável no sentido de que as classificações subsequentes feitas são ajustadas a favor das instâncias classificadas negativamente por classificações anteriores.

O AdaBoost é sensível ao ruído nos dados e casos isolados. Entretanto para alguns problemas é menos suscetível a perda da capacidade de generalização após o aprendizado de muitos padrões de treino (overfitting) do que a maioria dos algoritmos de aprendizado de máquina.

Ele combina múltiplos classificadores fracos para aumentar a curárica e assim obter um classificador forte.

O AdaBoost deve satisfazer 2 condições:
1. Ser treinado iterativamente em vários exemplo de treinmeanto ponderados
2. Cada interação tenta prover um excelente ajusate para eses exemplos por minimizar o erro de treinmento

**Prós**

**Contras**

### Código sklearn

https://chrisalbon.com/images/machine_learning_flashcards/AdaBoost_print.png

"The most important parameters are base_estimator, n_estimators, and learning_rate." (Adaboost Classifier, Chris Albon)

base_estimator: It is a weak learner used to train the model. It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
n_estimators: Number of weak learners to train iteratively.
learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.

----

Create Adaboost Classifier
The most important parameters are base_estimator, n_estimators, and learning_rate.

*base_estimator* is the learning algorithm to use to train the weak models. This will almost always not needed to be changed because by far the most common learner to use with AdaBoost is a decision tree – this parameter’s default argument.

n_estimators is the number of models to iteratively train.

learning_rate is the contribution of each model to the weights and defaults to 1. Reducing the learning rate will mean the weights will be increased or decreased to a small degree, forcing the model train slower (but sometimes resulting in better performance scores).

loss is exclusive to AdaBoostRegressor and sets the loss function to use when updating weights. This defaults to a linear loss function however can be changed to square or exponential.

algorithm{‘SAMME’, ‘SAMME.R’}, default=’SAMME.R’
If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

````
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target