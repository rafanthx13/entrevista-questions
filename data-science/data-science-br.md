---
layout: default
title: Data Science PT-BR
description: Questões de DataScience e Machine Learning
---

## Índex

## Links
+ [https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo](https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo)

## Conceituais

### DS-001 - O que é DataScience?

Ciência de dados é a atividade de extrair informação a apartir da análise de dados (estruturados (ml) ou não estruturados (dl)).

Usa-se técnicas da matemática, estatística e computação (algoritmos, ml e dl) para fazer isso.

Data Science NÃO É IA, Estatística, ML (Machine Learning), Big Data, Power BI nem Algoritmos.

### DS-002 - O que é Big Data?

Big data significa ter enorme volume de dados. O termo em si significa isso mas pode engloba toda a parte de arquitetura de uma empresa para suportar ter esse dados (como o Hadoop).

### DS-003 - Qual é a diferença entre ‘data science’ e ‘big data’?

Big Data em si não traz valor algum sem a técnica de Data Science. Então, Big Data é um objeto de análise de Data Science.

### DS-004 - Qual é a diferença entre um ‘data scientist’ e um ‘data analyst’?

A DataScience busca retirar informações apartir de técnicas computacionais: para Classificação, Regressão e etc..

Data Analytics resolve problemas de negócios, utilizando mais a estatística para resolver as coisas.

### DS-005 - Quais são os recursos fundamentais que representam big data?

Agora que abordamos as definições, podemos passar para as perguntas mais específicas de uma entrevista sobre data science. Tenha em mente, porém, que você será obrigado a responder perguntas relacionadas a data scientist, data analyst e big data. A razão para isso acontecer é porque todas essas subcategorias estão interligadas entre si.

Existem cinco categorias que representam big data e são chamadas de ” 5 Vs “:

+ Valor;
+ Variedade;
+ Velocidade;
+ Veracidade;
+ Volume.

Todos esses termos correspondem ao Big Data de uma maneira ou de outra.

### DS-006 - Qual a diferença entre IA, ML?

<img src="../img/difference-ia-ds-ml.png" />

## Técnicas 

### DS-007 - O que é Overfitting, Underfitting e Generalization, Bias e Variância?

### DS-008 - Quais técnicas utilizadas para tratamento de variáveis categóricas?

Label Encoding
+ Usada quando há poucos valores únicos categóricos
+ Mapeia cada valor para um Número
+ Atenção: Use-o para quando o valor categórico poder ser convertido numa representação numérica:
  - Exemplo: Ruim, Bom, Ótimo [1,2,3]
  - Pois os valores numéricos terão impacto na aprendizagem, pois vai considerar o valor 1 mais fraco que 3 (no exemplo acima)

One Hot Encoding
+ Usada quando a variável categórica tem diversos valores
+ Para cada valor único, cria-se uma coluna a mais. É colocado 0 ou 1 para o caso de ter aquele atributo.
+ Exemplo Um atributo 'cidade' de um estado
  - Se existir X cidades, então são criadas mais X features, para cada row do dataSet, somente uma dessa X novas features terá o valor 1, as outras X-1 features terão valor 0.

### DS-009 - O que é ‘validação cruzada’?

`cross-validation`: É executar um modelo de ML de várias formas diferentes (mudando a ordem de entrada das rows do dataset)

Ele verifica como determinados resultados de análises estatísticas específicas serão medidos quando colocados em um conjunto independente de dados.

### DS-010 - Qual é a diferença entre o aprendizado ‘supervisionado’ e ‘não supervisionado’?

Embora essa não seja uma das perguntas mais comuns das entrevistas, e, tenha mais a ver com machine learning do que com qualquer outra coisa, ela ainda assim pertence ao data science, portanto vale a pena saber a resposta.

Durante o aprendizado supervisionado, você infere uma função de uma parte rotulada de dados projetada para treinamento. Basicamente, a máquina aprenderia com os exemplos objetivos e concretos que você fornece.

Aprendizado não supervisionado refere-se a um método de treinamento de máquina que não usa respostas rotuladas – a máquina aprende por descrições dos dados de entrada.

### DS-011 - Diferencie Machine Learning (ML) de Deep Learning (DL)?

ML é uma área de IA que trata de algoritmos ou técnicas computacionais para máquinas/modelo/algoritmo aprender automaticamente com os dados.

DL é um conjunto de ML que trata das redes neurais com várias camadas e mais complexas.

### DS-012 - Qual a diferença entre Classificação e Regressão?

Os dois são atividades que podem ser realizadas por modelos de ML para predição.

A principal diferença é no valor de saída
+ Na classificação o valor é discreto (0 ou 1, ou [0,1,2,3..] ...)
+ Na regressão é contínuo (1,88; R$ 6.880,99 ...)

Em classificação, o objetivo é classificar uma row em determinada categoria.
+ Exemplo: Dado uma row com características de uma flor, classificar que tipo de planta é ela (Iris)

Em regressão, o objetivo é obter um valor numérico
+ Exemplo: Dado uma row de características de uma casa, predizer o valor dela  


### DS-013 - Explain what precision and recall are. How do they relate to the ROC curve?

https://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers.html/2

extrair tudo do link acima

The recall is alternatively called a true positive rate. It refers to the number of positives that have been claimed by your model compared to the number of positives that are available throughout the data.

Precision, which is alternatively called a positive predicted value, is based on prediction. It is a measurement of the number of accurate positives that the model has claimed as compared to the number of positives that the model has actually claimed.

### DS-014 - What is the difference between type I vs type II error?

https://www.datasciencecentral.com/profiles/blogs/understanding-type-i-and-type-ii-errors

“A type I error occurs when the null hypothesis is true, but is rejected. A type II error occurs when the null hypothesis is false, but erroneously fails to be rejected.”

Type I Error (False Positive Error)
A type I error occurs when the null hypothesisis true, but is rejected.  Let me say this again, atype I error occurs when the null hypothesis is actually true, but was rejected as falseby the testing.

A type I error, or false positive, is asserting something as true when it is actually false.  This false positive error is basically a "false alarm" – a result that indicates a given condition has been fulfilled when it actually has not been fulfilled (i.e., erroneously a positive result has been assumed).

Let’s use a shepherd and wolf example.  Let’s say that our null hypothesis is that there is “no wolf present.”  A type I error (or false positive) would be “crying wolf” when there is no wolf present. That is, the actual conditionwas that there was no wolf present; however, the shepherd wrongly indicated there was a wolf present by calling "Wolf! Wolf!”  This is a type I error or false positive error.

Type II Error (False Negative)
A type II error occurs when the null hypothesis is false, but erroneously fails to be rejected.  Let me say this again, atype II error occurs when the null hypothesis is actually false, but was accepted as trueby the testing.

A type II error, or false negative, is where a test result indicates that a condition failed, while it actually was successful.   A Type II error is committed when we fail to believe a true condition.

Continuing our shepherd and wolf example.  Again, our null hypothesis is that there is “no wolf present.”  A type II error (or false negative) would be doing nothing (not “crying wolf”) when there is actually a wolf present.  That is, the actual situationwas that there was a wolf present; however, the shepherd wrongly indicated there was no wolf present and continued to play Candy Crush on his iPhone.  This is a type II error or false negative error.

A tabular relationship between truthfulness/falseness of the null hypothesis and outcomes of the test can be seen in the table below:

### DS-015 -  O que é Regularização?

Técnica para tratar do problema de overfitting (quando o modelo se adapta demais aos dados de treinamento) ou de underfitting (quando não consegue se ajustar aos dados).

<img src="../img/overffiting-underfiting.png" />

A regularização coloca mais informação para dar penalidade aos dados que trariam a condição de overfitting/underfitting.

Ele ajuda a reduzir a complexidade do modelo e assim fazer melhores previsões.

É aconselhado em que tem: poucas features para um dataSet muito grande ou ao contrário, quando há muitas features para poucos dados.

Ver L1, L2, Lasso, Ridge.

link: http://enhancedatascience.com/2017/07/04/machine-learning-explained-regularization/

### DS-001 - Como avaliar modelos de ML?

Separa os dados em *train* e *test* de forma aleatória. Aplica o modelo na base de treinamento e avalia o seu modelo na base de teste.

Pode-se usar a técnica de *cross-validation* para garantir que essa divisão é adequada.

Pode-se avaliar pelos seguintes critérios
+ Matriz de Confusão
+ Curva ROC, CAP e seus respectivos AUC
+ Acurácia, Precisão, Recall, Sensitividade, Especifidade
+ F1 Score

https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b

### DS-001 - O que fazer com dados corrompidos ou faltantes?

Podemos:
+ Retirar as rows (se forem poucas estiverem corrompidas)
+ Retirar as colunas (se muitas rows estiverem corrompidas)
+ Colocar a média ou algum valor que faça sentido.

Dissecar:
https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/


### DS-001 - Como selecionar as features mais importantes de um DataSet?

+ Remove as features que estão correlacionadas

Answer: Following are the methods of variable selection you can use:

Remove the correlated variables prior to selecting important variables
Use linear regression and select variables based on p values
Use Forward Selection, Backward Selection, Stepwise Selection
Use Random Forest, Xgboost and plot variable importance chart
Use Lasso Regression
Measure information gain for the available set of features and select top n features accordingly.
 

### DS-001 - Q19. What is the difference between covariance and correlation?

> Answer: Correlation is the standardized form of covariance.

>Covariances are difficult to compare. For example: if we calculate the covariances of salary ($) and age (years), we’ll get different 
>covariances which can’t be compared because of having unequal scales. To combat such situation, we calculate correlation to get a value 
>between -1 and 1, irrespective of their respective scale.

 

### DS-001 - Q20. Is it possible capture the correlation between continuous and categorical variable? If yes, how?

Answer: Yes, we can use ANCOVA (analysis of covariance) technique to capture association between continuous and categorical variables.

