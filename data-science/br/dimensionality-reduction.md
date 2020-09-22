---
layout: default
title: Redução de Dimensionalidade PT-BR
description: Questões de DataScience e Machine Learning
---

# Redução de Dimensionalidade e técnicas

A redução de dimensionalidade pode ser uma boa técnica visual para provar que os dados de classes diferentes são passíveis de serem separáveis, ou seja, aplicando uma técnica de redução e em seguida mostrando clusters em gráficos, se você consegue separar os dados de cada classe, entoa, a classificação deverá ser possível

## Técnicas

### Técnicas de Redução de dimensionalidade

+ SVD
  - Singular Value Decomposition - Decomposição de valores singulares
+ PCA
  - Principal Component Analysis - Análise de Componentes Principais
+ LDA
  - Análise Discriminante Linear - Linear Discriminant Analysis
+ t-SNE
  - Incorporação Estocástica de Vizinhos com Distribuição de T- t-Distributed Stochastic Neighbour Embedding
+ Autoencoders
+ Transformadas de Fourier e Wavelet

### LDA

**LDA (Linear Discriminant Analysis)**

Possui mesma função do PCA mas envolve a classe dos dados, ou seja, é um algoritmo supervisionado.

=> Para problemas linearmente separáveis

• Além de encontrar os componentes principais, LDA também encontra os eixos que maximizam a separação entre múltiplas classes

• É um algoritmo supervisionado por causa da relação que tem com as classes

• Das m variáveis independentes, LDA extrai p <= m novas variáveis independentes que mais separam as classes da variável dependente

**Diferença entre PCA e LDA**

Uma maneira simples de visualizar a diferença entre o PCA e o LDA é que o PCA trata todo o conjunto de dados como um todo, enquanto o LDA tenta modelar as diferenças entre as classes dentro dos dados. Além disso, o PCA extrai alguns componentes que explicam mais a variância, enquanto o LDA extrai alguns componentes que maximizam a separabilidade de classes.

**Podemos usar LDA para regressão?**

LDA é Análise Discriminante Linear. É uma generalização do discriminante linear de Fisher, um método usado em estatística, reconhecimento de padrões e aprendizado de máquina para encontrar uma combinação linear de recursos que caracterizam ou separam duas ou mais classes de objetos ou eventos. A combinação resultante pode ser usada como um classificador linear ou, mais comumente, para redução de dimensionalidade antes da classificação posterior. No entanto, para a regressão, temos que usar ANOVA, uma variação do LDA. A LDA também está intimamente relacionada à análise de componentes principais (PCA) e à análise fatorial, pois ambas procuram combinações lineares de variáveis ​​que melhor explicam os dados. O LDA tenta explicitamente modelar a diferença entre as classes de dados. O PCA, por outro lado, não leva em consideração nenhuma diferença de classe e a análise fatorial constrói as combinações de recursos com base nas diferenças em vez de semelhanças. A análise discriminante também é diferente da análise fatorial por não ser uma técnica de interdependência: uma distinção entre variáveis ​​independentes e variáveis ​​dependentes (também chamadas de variáveis ​​de critério) deve ser feita. O LDA funciona quando as medições feitas nas variáveis ​​independentes para cada observação são quantidades contínuas. Ao lidar com variáveis ​​independentes categóricas, a técnica equivalente é a análise de correspondência discriminante.

### SVD

Decomposição de valor singular

What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️

Singular Value Decomposition (SVD) is a general matrix decomposition method that factors a matrix X into three matrices L (left singular values), Σ (diagonal matrix) and R^T (right singular values).
For machine learning, Principal Component Analysis (PCA) is typically used. It is a special type of SVD where the singular values correspond to the eigenvectors and the values of the diagonal matrix are the squares of the eigenvalues. We use these features as they are statistically descriptive.
Having calculated the eigenvectors and eigenvalues, we can use the Kaiser-Guttman criterion, a scree plot or the proportion of explained variance to determine the principal components (i.e. the final dimensionality) that are useful for dimensionality reduction.

Decomposição de valores singulares (SVD) é um método geral de decomposição de matrizes que fatora uma matriz X em três matrizes L (valores singulares à esquerda), Σ (matriz diagonal) e R^T (valores singulares à direita).

Para aprendizado de máquina, a Análise de Componentes Principais (PCA) é normalmente usada. É um tipo especial de SVD onde os valores singulares correspondem aos autovetores e os valores da matriz diagonal são os quadrados dos autovalores. Usamos esses recursos porque são estatisticamente descritivos.

Tendo calculado os autovetores e autovalores, podemos usar o critério de Kaiser-Guttman, um gráfico ou a proporção da variância explicada para determinar os componentes principais (ou seja, a dimensionalidade final) que são úteis para a redução da dimensionalidade.

### t-SNE

t-sne (se fala ti-isni) é uma técnica para reduzir a dimensionalidade preservando a separação de clusters

https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
https://www.datacamp.com/community/tutorials/introduction-t-sne?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034358&utm_targetid=aud-299261629574:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1001520&gclid=CjwKCAjw1K75BRAEEiwAd41h1PJ-mOJFa1k75ThPZd7d6BjS6itByrtS4fM95X_8-atp4MYPq_63khoCE1IQAvD_BwE

Para entender esse algoritmo, você precisa entender os seguintes termos:
+ Distância euclidiana
+ Probabilidade Condicional
+ Gráficos normais e de distribuição T

https://www.youtube.com/watch?v=NEaUSP4YerM

O algoritmo t-SNE pode agrupar com precisão os casos de fraude e não fraude em nosso conjunto de dados.

Embora a sub-amostra seja bastante pequena, o algoritmo t-SNE é capaz de detectar clusters com muita precisão em todos os cenários (embaralhe o conjunto de dados antes de executar o t-SNE).

Isso nos dá uma indicação de que outros modelos preditivos terão um bom desempenho na separação de casos de fraude de casos não fraudulentos.

t-Distributed Stochastic Neighbour Embedding (t-SNE, pronunciado tí-ciní) é uma técnica de visualização em 1D, 2D ou 3D de datasets de altas dimensões.

O t-SNE, desenvolvido por Laurens van der Maaten e Geoffrey Hinton, é um refinamento do SNE e se diferencia deste, principalmente, pela utilização de distribuição t de Student para representar os dados em baixas dimensões (ou dados mapeados).

O t-SNE utiliza um kernel gaussiano para converter pontos em altas dimensões para probabilidades de conexões (P) e um kernel t-Student, com um grau de liberdade, para representar as probabilidades de conexões entre os pontos em baixas dimensões (Q), no espaço mapeado. O custo das diferenças entre as duas distribuições P e Q é modelado pela divergência de Kullbach-Leibler20, o gradiente desta função (ver [Equação 5] abaixo) é utilizado para atualizar o mapa de pontos em baixas dimensões.
 

## PCA

PCA Intuition
What is the true purpose of PCA?
The true purpose is mainly to decrease the complexity of the model. It is to simplify the model while keeping relevance and performance. Sometimes you can have datasets with hundreds of features so in that case you just want to extract much fewer independent variables that explain the most the variance.

What is the difference between PCA and Factor Analysis?
Principal component analysis involves extracting linear composites of observed variables. Factor analysis is based on a formal model predicting observed variables from theoretical latent factors. PCA is meant to maximize the total variance to look for distinguishable patterns, and Factor analysis looks to maximize the shared variance for latent constructs or variables.

Should I apply PCA if my dataset has categorical variables?
You could try PCA, but I would be really careful, because categorical values can have high variances by default and will usually be unstable to matrix inversion.

Apply PCA and do cross validation to see if it can generalize better than the actual data. If it does, then PCA is good for your model. (Your training matrix is numerically stable). However, I am certain that in most cases, PCA does not work well in datasets that only contain categorical data. Vanilla PCA is designed based on capturing the covariance in continuous variables. There are other data reduction methods you can try to compress the data like multiple correspondence analysis and categorical PCA etc.

What is the best extra resource on PCA?
Check out this video that has an amazing explanation of PCA and studies it in more depth.

---

Significa: "Análise de Componentes principais"

=> Para problemas linearmente separáveis

=> Deve-se ser feito asobre dados numéricos

**Diferenciar Seleção de Extração de características**

• Seleção de características x Extração de características
+ Seleção: Indicar os atributos mais importantes
+ Extração: AO fazer uma análise, criar novos atributos. É como unir atributos.
  - Encontrar relacionamentos entre os atributos para combinar eles e reduzir a dimensionalidade

Lembre-se, o PCA não é escolher os melhore "n" atributos, e sim reduzir para "n" atributos

**Características do PCA**

• PCA: Identifica a correlação entre variáveis, e caso haja uma forte
correlação é possível reduzir a dimensionalidade.
  - Exemplo: Se vocÊ têm duas variáveis com forte correlação você pode então unilas e assim reduzir em 1 a quantidade de features.

**Funcionamento**

• Um dos principais algoritmos de aprendizagem de máquina não
supervisionada (não há certo/errado a priori)
• Das m variáveis independentes, PCA extrai p <= m novas variáveis
independentes que explica melhor a variação na base de dados, sem
considerar a variável dependente
• O usuário pode escolher o número de p

COMPONETSE PRICINCPAIS são OS COMEPONETNE DE features concatenados.

**Se com PCA ficar pior?**

Abaixou um pouco, mas, você deve avaliar o `trade_off` entre a precisão e a velocidade.

Exemplo, será que mesmo reduzindo 1\% poderia ser melhor usar o PCA pois teria menos dados para classificar e assim ter menos custo computacional?

## Outras

### Qual é a maldição da dimensionalidade? Por que nos preocupamos com isso? ‍

Os dados em apenas uma dimensão são compactados de maneira relativamente compacta. Adicionar uma dimensão alonga os pontos dessa dimensão, afastando-os ainda mais. Dimensões adicionais espalham os dados ainda mais, tornando os dados de alta dimensão extremamente esparsos. Nós nos preocupamos com isso, porque é difícil usar o aprendizado de máquina em espaços escassos.