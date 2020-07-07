---
layout: default
title: Data Science PT-BR
description: Questões de DataScience e Machine Learning
---

## Índex

## Links
+ [https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo](https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo)

## Conceituais e Teóricas

### DS-001 - O que é Data Science?

Ciência de dados é a atividade de extrair informação a apartir de dados, estruturados (usando ml) ou não estruturados ( usando dl).

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

### DS-006 - Qual a diferença entre IA, ML e DL?

ML, DL e NN (Neural Networks) são subconjuntos da área de INteligência Artificial

IA < ML < DL 

<img src="../img/difference-ia-ds-ml.png" />

ML é uma área de IA que trata de algoritmos ou técnicas computacionais para máquinas/modelo/algoritmo aprender automaticamente com os dados.

DL é um conjunto de ML que trata das redes neurais com várias camadas e mais complexas.

---
---
---

## + Técnicas 

### DS-007 - O que é Overfitting, Underfitting e Generalization, Bias e Variância?

<img src="../img/overffiting-underfiting.png" />

<img src="../img/bias-variance.png" />

+ **High Variance:** Baixo erro em dados de treino e alto erro em dados de teste
+ **High Bias:** Alto erro em dados de treino e erro parecido em dados de teste
+ **High Bias and Variance**: Alto erro em dados de treino e erro maior em dados de teste
+ **Low Bias and Variance:** Baixo erro em dados de treino e baixo erro em dados de teste

**What is low/high bias/variance?**

These concepts are important to understand k-Fold Cross Validation:
Low Bias is when your model predictions are very close to the real values.
High Bias is when your model predictions are far from the real values.
Low Variance: when you run your model several times, the different predictions of your observation points
won’t vary much.
High Variance: when you run your model several times, the different predictions of your observation points
will vary a lot.

### DS-010 - Qual é a diferença entre o aprendizado ‘supervisionado’ e ‘não supervisionado’?

Embora essa não seja uma das perguntas mais comuns das entrevistas, e, tenha mais a ver com machine learning do que com qualquer outra coisa, ela ainda assim pertence ao data science, portanto vale a pena saber a resposta.

Durante o aprendizado supervisionado, você infere uma função de uma parte rotulada de dados projetada para treinamento. Basicamente, a máquina aprenderia com os exemplos objetivos e concretos que você fornece.

Aprendizado não supervisionado refere-se a um método de treinamento de máquina que não usa respostas rotuladas – a máquina aprende por descrições dos dados de entrada.

### DS-012 - Qual a diferença entre Classificação e Regressão?

Os dois são atividades que podem ser realizadas por modelos de ML para predição.

A principal diferença é no valor de saída
+ Na classificação o valor é discreto (0 ou 1, ou [0,1,2,3..] ...)
+ Na regressão é contínuo (1,88; R$ 6.880,99 ...)

Em classificação, o objetivo é classificar uma row em determinada categoria.
+ Exemplo: Dado uma row com características de uma flor, classificar que tipo de planta é ela (Iris)

Em regressão, o objetivo é obter um valor numérico
+ Exemplo: Dado uma row de características de uma casa, predizer o valor dela  


### DS-001 Qual a diferença entre covariância e correlação?

Correlação é a forma padronizada de covariância.

Covariâncias são difíceis de comparar. Por exemplo: se calcularmos as covariâncias de salário ($) e idade (anos), teremos diferentes
covariâncias que não podem ser comparadas por causa de escalas desiguais.

Para combater essa situação, calculamos a correlação para obter um valor
entre -1 e 1, independentemente da respectiva escala.
 

### DS-001 - É possível capturar a correlação entre variável contínua e variável categórica? Se sim, como?

Sim, podemos usar a técnica ANCOVA (análise de covariância) para capturar a associação entre variáveis contínuas e categóricas.

O acrônimo ANCOVA vem de “ANalysis of COVAriance”. Na realidade, a ANCOVA combina dois tipos de estratégias: Análise de Variância (ANOVA) e Análise de Regressão.

A análise de covariância permite aumentar a precisão dos experimentos e eliminar os efeitos de variáveis ​​que nada têm a ver com o tratamento , mas que, no entanto, estão influenciando os resultados.

Além disso, permite obter mais informações sobre a natureza dos tratamentos que estamos aplicando em nossa pesquisa. Em resumo, nos ajuda a ajustar nossos resultados para torná-los mais confiáveis.

https://pt.slideshare.net/UbirajaraFernandes/ancova-anlise-de-covarincia-ecologia-quantitativa-ubirajara-l-fernandes

https://maestrovirtuale.com/analise-de-covariancia-ancova-o-que-e-e-como-e-usado-em-estatistica/

---
---
---

## Regressão e Regularização

### O que é regressão? Que modelos você pode usar para resolver problemas de regressão

A regressão faz parte da área  de aprendizagem supervisionado de ML. Os modelos de regressão investigam a relação entre uma variável (s) dependente (*target*) independente (s) (*features*).

Exemplos

+ **Regressão linear** estabelece uma relação linear entre alvo e preditor (es). Ele prevê um valor numérico e tem o formato de uma linha **reta**.
+ **Regressão polinomial** tem uma equação de regressão com o poder da variável independente maior que 1. É uma **curva** que se encaixa nos pontos de dados.
+ **Regressão de Ridge** ajuda quando preditores são altamente correlacionados (problema de multicolinearidade). Ele penaliza os quadrados dos coeficientes de regressão, mas não permite que os coeficientes atinjam zeros (usa a regularização L2).
+ **Regressão do lasso** penaliza os valores absolutos dos coeficientes de regressão e permite que alguns deles alcancem o zero absoluto (permitindo a seleção de recursos). Usa regularização L1.
+ **Regressão Elastic-Net**: Usa as regularização L1 e L2.

### Quais métricas para avaliar modelos de regressão você conhece?

<img src="../img/metricas-erros-01.png" />

**Mean Squared Error(MSE) | Erro quadrático médio (EQM)**
+ Média da Somatória da diferença entre valor esperado (y) e valor previsto (ŷ) elevado ao quadrado
+ O MSE eleva o quadrado por duas razões:
  1. Erro acima do valor real ou abaixo vão ficar positivos (pois eleva ao quadrado.
  2. Os módulos dos erros maiores vão gerar maior penalidade, assim tende a ser mais impactado por outliers.
+ Tanto MSE quanto RMSE são muito impactados pela presença de outliers no Y. 
  - Então, se com essa métrica parecer ruim, observe se há ou não outliers pois eles podem está atrapalhando a sua métrica
**Root Mean Squared Error(RMSE) | Raiz do Erro Quadrático Médio (REQM**
+ Raiz da MSE, dessa forma volta a dimensão anterior dos erros antes de serem elevados ao quadrado


**(R)MSLE - (Root) Mean Squared Logarithmic Error - Raiz Quadrada do Erro Médio Logarítmico Quadrado**
+ É o MSE mas aplicando um Log
+ Ele acaba sendo uma aproximação do MSE para um **Erro percentual**
+ Matematicamente ele é mais fácil de minimizar
+ O MSE se importa com a diferença "absoluta", enquanto que o MSLE se importa com a diferença "relativa" por calcular o erro como uma *diferença percentual* entre o valor real (y) e o previsto (ŷ)
+ [link +msle](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle))
+ Costuma ser usado em vendas, exemplo: se eseprar vender 1000 e vende 1001 o impcato é menor pois só o fato de conseguir prever 1000 já é um grande ganho. Se usa-semos MSE o impcato entre y = 2 ŷ = 3 e seria o mesmo que y = 1000 e ŷ = 10001, que para uma venda, nâo é impcatante assim.
+ **Comparando a questão de OutLiers entre MSE e MSLE**
  - O MSLE é sensível a outliers, bem menos que MSE porque há diferença que ele faz é relativa

<img src="../img/msle-01.png" />

**Mean Absolute Error(MAE) | Erro Absoluto Médio (EAM)**
+ Somatório da diferença entre o valor esperado (y) e o valor previsto (ŷ) dividido pela quantidade de previsões
+ Não é sensitivo a outilers, por isso, é usada para quando não se quer que os outilers tenham impacato na avaliação
+ Por isso você usa quando não tem outilers em geral, quando sâo extremamente raros
**Observação: MAE e MSE**
+ MSE e RMSE penalizam outliers e o MAE não
+ Então, se ao analisar seus dataset, os outliers existirem mas forem realmente parte dos seus dados, então, é recomendável usar MSE. 
+ Agora, se puder retirar os outliers, então é melhor o MAE


**MedAE - Median Absolute Error - Erro Mediano Absoluto**
+ As vezes também chamado de MAD
+ Fórmula: É a mediana da Serie dos módulos dos erros abolutos para cada predição
  - 1. Forma uma lista dos erros absolutos; 2. Aplica módulo (todos ficam positivos); 3. busca a mediana
+ Quanto menor, melhor
- Esse erro é raro de ver sendo usado, mas o foi no site de preço de casas na [Zillow](https://www.zillow.com/research/putting-accuracy-in-context-3255/)
+ MedAE é uma medida robusta , similar ao MAE por ignorar muito outilers, pois, se o erro nos outilers forem muito grandes, eles vão para a cabeça(head) ou calda (tails) da lista e assim só vai mover o index da median uma casa pra frente/trás da listagem de erros absolutos.

Lembrando: Mediana é o valor que fica no meio caso você ordenar tudo em forma ascendente um conjunto de dados.

Exemplo de como funciona
````python
import numpy as np
from sklearn.metrics import median_absolute_error

y_true = np.asarray([3, -0.5, 2, 7, 6])
y_pred = np.asarray([2.5, 0.0, 2, 8, 9])

print(np.sort(np.absolute(y_true - y_pred)))
# [0.  0.5 0.5 1.  3. ] Ordenando os erros em valores absolutos, o valor do meio é a MedAE: 0.5

median_error_manual = np.median(np.absolute(y_true - y_pred)) # calculo manual
print("manual", median_error_manual, "| sklearn", median_absolute_error(y_true, y_pred))
# manual 0.5 | sklearn 0.5
````

<img src="../img/maed.png" />

**MAPE - Mean Absolute Percentage Error - Erro Médio Percentual Absoluto**
+ Erro mais fácil de ser explicado, fica entre [0,1], quanto menor, melhor
+ É a média das porcentagens de erro
+ No exemplo abaixo temos se o  de MAPE é 0.42, quer dizer que: **O modelo erra 42% em média** em mas nâo se sabe se é acima do valor ou abaixo do valor real.


**R² or Coefficient of Determination | **Coeficiente de Determinação** $ R^2 $. **
+ [Porque o R2 é inútil](https://data.library.virginia.edu/is-r-squared-useless/)
+ Basicamente, este coeficiente R² indica quanto o modelo foi capaz de explicar os dados coletados. O R² varia entre 0 e 1, indicando, em percentagem, o quanto o modelo consegue explicar os valores observados. Quanto maior o R², mais explicativo é o modelo, melhor ele se ajusta à amostra.
+ Por exemplo, se o R² de um modelo é 0,8234, isto significa que 82,34% da variável dependente consegue ser explicada pelos regressores presentes no modelo.
+ O R² deve ser usado com precaução, pois é sempre possível torná-lo maior pela adição de um número suficiente de termos ao modelo. Assim, se, por exemplo, não há dados repetidos (mais do que um valor `y` para um mesmo `x` ) um polinômio de grau `n - 1` dará um ajuste perfeito R² = 1 para n  dados. Quando há valores repetidos, o R² não será nunca igual a 1, pois o modelo não poderá explicar a variabilidade devido ao erro puro.
r. 



### O que é Regularização?

Técnica para tratar do problema de overfitting (quando o modelo se adapta demais aos dados de treinamento) ou de underfitting (quando não consegue se ajustar aos dados).

<img src="../img/overffiting-underfiting.png" />

A regularização coloca mais informação para dar penalidade aos dados que trariam a condição de overfitting/underfitting.

Ele ajuda a reduzir a complexidade do modelo e assim fazer melhores previsões.

É aconselhado em que tem: poucas features para um dataSet muito grande ou ao contrário, quando há muitas features para poucos dados.

### Que tipo de técnicas de regularização são aplicáveis aos modelos lineares?

Regularização L1 (regularização Lasso) - Adiciona a soma dos valores absolutos dos coeficientes à função de custo.
Regularização L2 (regularização Ridge) - Adiciona a soma dos quadrados dos coeficientes à função de custo.

Há outro que são: AIC/BIC, Ridge, Lasso, Basis pursuit denoising, Rudin–Osher–Fatemi model (TV), Potts model, RLAD, Dantzig Selector,SLOPE

### Podemos usar a regularização L1 para a seleção de features?

Sim, porque a natureza da regularização L1 (Lasso) levará a coeficientes com pouco valor à zero

Exemplo de Lasso
https://towardsdatascience.com/feature-selection-using-regularisation-a3678b71e499
````python
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)
sel_.get_support()

# sel_.get_support(): Mostrará uma matriz de True/False, onde False serão as variáveis que foram levadas a Zero

# A seguir, selecionamos as colunas com True
selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))
````

### Quando a regressão de Ridge é favorável em relação à regressão de Lasso?

Na presença de poucas variáveis com um dataset de tamanho médio / grande, use a regressão Lasso. 

Na presença de muitas variáveis com efeito de tamanho pequeno / médio, use regressão Ridge.

Conceitualmente, podemos dizer que a regressão de laço (L1) faz seleção de variáveis e encolhimento de parâmetros, enquanto a regressão de Ridge apenas encolhe e acaba incluindo todos os coeficientes do modelo. Na presença de variáveis correlacionadas, a regressão de Ridge pode ser a escolha preferida. Além disso, a regressão de Ridge funciona melhor em situações em que as estimativas menos quadradas têm maior variação. Portanto, depende do objetivo do nosso modelo.

### Quando a regularização se torna necessária no Machine Learning?

https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/

A regularização torna-se necessária quando o modelo começa a adequar demais (overfitting) ou não se adequar (underfitting). Essa técnica introduz um termo de custo para trazer mais recursos com a função objetivo. Portanto, ele tenta empurrar os coeficientes de muitas variáveis para zero e, portanto, reduzir o custo. Isso ajuda a reduzir a complexidade do modelo para que o modelo possa se tornar melhor na previsão (generalização).

---
---
---

## Avaliar Modelos

### Cross-Validattion

#### train_test_split ou holdout strategy

[sklearn - cross validation](https://scikit-learn.org/stable/modules/cross_validation.html)

<div style="text-align: center;">
	<img src="../img/holdout-strategy.jpg" />	
</div>

Pros of the hold-out strategy: Fully independent data; only needs to be run once so has lower computational costs.

Cons of the hold-out strategy: Performance evaluation is subject to higher variance given the smaller size of the data.

**Problema doldout**

AO dividir em train/test, podemos cair no seguinte erro. Tunar os parâmetros para que obtenham o menor erro na base de teste. Perceba que a tunagem está diretamente ligada a essa base de test criando assim um overfiting (nâo consegue generalizar os dados).

Uma forma de evitar isso é fazer mais uma divisão da base, em uma de validaçâo (`validation`).

Essa é testada somente no final. Todo o ajuste de erro é feito olhando para a de test e não apra essa nova.


---

Prém vai acontecer o seguinte problea: dividir em 3 a base vai ter poucos dados para fazer o train/test/validate. 

Entâo, utilizamos cross-validation (apelido : CV) para fazer um treinamento otimizado com essa base menor, além de resolver outro problemas.

CV é uma técnica: em ter esse 3 datasets: train/test/final_validation e pode ser feito de várias formas

#### K-Fold CV

[link kdnuggets](https://www.kdnuggets.com/2017/08/dataiku-predictive-model-holdout-cross-validation.html)

K-fold validation evaluates the data across the entire training set, but it does so by dividing the training set into K folds – or subsections – (where K is a positive integer) and then training the model K times, each time leaving a different fold out of the training data and using it instead as a validation set. At the end, the performance metric (e.g. accuracy, ROC, etc. — choose the best one for your needs) is averaged across all K tests. Lastly, as before, once the best parameter combination has been found, the model is retrained on the full data.

Pros of the K-fold strategy: Prone to less variation because it uses the entire training set.

Cons of the K-fold strategy: Higher computational costs; the model needs to be trained K times at the validation step (plus one more at the test step).

<div style="text-align: center;">
	<img src="../img/validation-04.png" />	
</div>

<div style="text-align: center;">
	<img src="../img/kfold-strategy.jpg" />	
</div>

### Como avaliar modelos de ML, para classificação e regressão?

Separa os dados em *train* e *test* de forma aleatória. Aplica o modelo na base de treinamento e avalia o seu modelo na base de teste.

Pode-se usar a técnica de *cross-validation* para garantir que essa divisão é adequada.

Os critérios de avaliação são válidos para determinadas atividade de ML

Classificação (y discreto)
+ Matriz de Confusão
+ Curva ROC, CAP e seus respectivos AUC
+ Acurácia, Precisão, Recall, Sensitividade, Especifidade
+ F1 Score

Regressão (y contínuo)
+ MAE, MSE, RMSE
+ R²

https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b

### Por que precisamos dividir nossos dados em três partes: treinamento, validação e teste?

O conjunto de treinamento é usado para ajustar o modelo, ou seja, para treinar o modelo com os dados.

O conjunto de validação é então usado para fornecer uma avaliação imparcial de um modelo enquanto o ajuste dos hiper parâmetros é feito. Isso melhora a generalização do modelo. 

Finalmente, um conjunto de dados de teste que o modelo nunca “viu” antes deve ser usado para a avaliação final do modelo. Isso permite uma avaliação imparcial do modelo. A avaliação nunca deve ser realizada com os mesmos dados usados para o treinamento. Caso contrário, o desempenho do modelo não seria representativo.

## O que é a Matriz de confusão e tudo o que ele engloba?

**Acurácia, Recall, Sensibilidade, F1 Score, Erro tipo 1 e 2**

Exemplo para uma classificação binária temos a seguinte matriz de confusão:

````
Matrix de Confusão    Nomenclatura das partes
 [ 65  3  ]               [ TP  FP ]
 [  3  29 ]               [ FN  TN ]
````

Onde:
+ TP (True Positive)  | Verdadeiro Positivo : A Classe é 1 e Previu 1 (O modelo acertou)
+ FP (False Positive) | Falso Positivo : A Classe é 1 e Previu 0 (O modelo errou) (erro tipo 1)
+ FN (False Negative) | Falso Negativo : A Classe é 0 e Previu 1 (O modelo errou) (erro tipo 2)
+ TN (True Negative)  | Verdadeiro Negativo : A Classe é 0 e Previu 0 (O modelo Acertou)

<img src="../img/metricas-erros-02.png" />

#### Erro Tipo 1 e tipo 2

**Erro tipo 1 / Taxa de Falso Positivo | False Positive Rate**
+ Quando afirma que pertence a uma classe quando na verdade não pertence
+ Erro Tipo 1 = FP/(FP + TN)

**Erro Tipo 2 / Taxa de Falso Negativo | False Negative Rate**
+ Quando afirma que não pertence a uma classe mas na verdade pertence
+ Erro Tipo 2 = FN/(FN + TN)

<div style="text-align: center;">
	<img src="../img/metricas-erros-03.jpg" style="width: 70%; align-content: center" />	
</div>

**Acurácia / Accuracy**

Métrica possível para avaliar o modelo de classificação para todas as classes.

Acurácia é a porcentagem total dos itens classificados **corretamente**

<div style="text-align: center;">
	<img src="../img/metricas-erros-03.png" />
</div>

Porém, em um dataset desbalanceado ela não será uma métrica tão boa.

**Observação: Nem sempre acurácia é uma boa métrica**

A Acurrácia não é uma boa métrica  quando há o dataSet está desbalanceado na quantidade de registros por classe. Por exemplo, na classificação binária com 95% da classe A e 5% da classe B, a precisão da previsão pode ser de 95%. Em datasets desbalanceados, precisamos escolher Precisão, Recall ou F1 Score, dependendo do problema que estamos tentando resolver.

- Definiçâo de Acurácia: **Porcentagme de acerto do modelo**
- não use "oficialmente" (como métrica final a apresentar), apenas "preguiçosamente", há coisas muito melhores
- inadequada para dados desequilibrados, pode te enganar
- Exemplo: Imagine que você vai fazer um detector de spam. Na sua caixa de email hoje, cerca de 98% dos seus emails nâo são span. Por causa disso, se você simplismente atribuir todos os emails como não-spam, você consegue uma acurácia monstruosa de 98% sem ser capaz de detectar um único spam. Isso acaontece porque a qtd de spam é extremamente baixa em realaçâo a qtd de não-spam, ou seja, seu dataset está desbalanceado.

<img src="../img/metricas-erros-04.png" />

#### Precisão, Precision

É a taxa da quantidade de itens positivos que foram devidamente classificados como positivos, ou seja, a taxa de acerto para classificar os itens de uma classe.

Precisão = TP / (TP + FP)
<div style="text-align: center;">
<img src="../img/metricas-erros-06.png" />
</div>

- Definição de Precisão: **Dos casos que eu previ como positivos (para uma classe) quantos realmente são?**
- Envio de cupons de desconto, custos diferentes para cada erro.
- Ex: se custa caro mandar a promoção, das pessoas que eu previ que iam comprar, quantas compraram?

#### Sensitividade, Recall, hit rate TPR (True Positive Rate)

Taxa de itens positivos a uma classe, que fora classificados como positivo pelo modelo do total de itens positivos.

Recall = TP / (TP + FN)

<!--
The recall is alternatively called a true positive rate. It refers to the number of positives that have been claimed by your model compared to the number of positives that are available throughout the data.
-->

<div style="text-align: center;">
	<img src="../img/metricas-erros-05.png" />
</div>

- Definição de Recall: dos que eram realmente positivos (para uma classe) quantos eu detectei?
- Chamado de taxa de detecção

#### Especificidade, Seletividade Specifity, TNR (True Negative Rate)

Taxa de itens previstos como não pertencente a classe do total desses itens negativos a essa classe.

Especificidade = TN/ ( TN + FP)
<div style="text-align: center;">
<img src="../img/metricas-erros-08.png" />
</div>

#### F1 Score

É a média harmônica entre precisão e recall. É uma medida melhor que o da acurácia quando as classe do dataSet estão desbalanceada pois ela vai refletir esse desbalanceamento.

F1 Score = 2 * Precisão * Recall / ( Precisão + Recall)

A média harmônica captura quando a quantidade de registros de uma classe é maior do que outra.

Exemplo:

<div style="text-align: center;">
<img src="../img/metricas-erros-11.png" />
</div>

#### Matrix de confusão em scikit-learnig

Exemplo: Para um dataSet com dados de um possível cliente para comprar ou não um produto temos

````python
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
precisao = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)
print("Accuracy\n", precisao, "\n")
print("Matrix de Confusao\n",matriz, "\n")
print("Matrix de Confusao Porcentagem\n",matriz/matriz.sum(), "\n")
print(classification_report(y_test,y_pred, target_names=['Not Purchased', 'Purchased']))
````
Gerando

````python
Accuracy
 0.94 

Confusion Matrix
 [[65  3]
 [ 3 29]] 

Confusion Matrix Percentage
 [[0.65 0.03]
 [0.03 0.29]] 

               precision    recall  f1-score   support

Not Purchased       0.96      0.96      0.96        68
    Purchased       0.91      0.91      0.91        32

     accuracy                           0.94       100

Avaliando a classificacao para cada classe:
+ Precision: Numero total de positivos do total dos classificados como positivos para uma classe
+ Recall: Numero de Positivos que foram devidamente identificados para uma classe
+ F1-Score: Media Harmonica entre precisao e recall
+ Support: Quantidade de amostras de uma classe

Avaliacao geral
+ Acurracia
````

### Kappa
+ Mede a concordância entre seu modelo e um modelo aleartório
+ Uma boa métrica que pouca gente conhece.
+ Costuma-se usar ele em multi-classes
+ Considera-se geralmente uma medida mais robusta do que o simples cálculo percentual de concordância, pois κ leva em consideração a possibilidade da ocorrência de um acaso
- [https://en.wikipedia.org/wiki/Cohen%27s_kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
+ Interpretando
  - Quanto maior o valor kappa, melhor
  - Ele pode ser negativo

````python
 from sklearn.metrics import cohen_kappa_score

print("P = {}\nY = {}".format(p_multi_argmax, y_multi))
# P = [2 1 0 1 2 1 2 0 0 1]
# Y = [1 1 1 1 2 2 0 0 1 0]
cohen_kappa_score(y_multi, p_multi_argmax)
# 0.07692307692307687
````

### Log Loss or cross-entropy loss
+ **INDEPENDE DO Threshdold/pnto de corte**
- calculada para a probabilidade empírica do evento. Proporção que o evento ocorre na vida real
- Se o time A jogar contra o time B e tiver 40% de chances de ganhar, se jogarem 10 vezes, 4 vezes o time A vai ganhar.
- Se tivermos um modelo para prever isso, entâão, A log loss estará na mínima quando o modelo prever 0.4
- Ou seja, nosso modleo atingir 0.4 significa que está ótimo

**Se um evento no mundo real tem uma probabilidade limitada de acontecer, então nosso modelo também deverá ter essa mesma probabilidade na log loss se for perfeito**

Em um evento onde já se sabe a probaiblidade, já sabemos o limite que uma log loss pode ter, então, quão mais próximo dessa porcentagem melhor o nosos modelo.

A log loss estará minimizada (loss é o erro, erro mínimo == melhor modelo) quando o modelo prever exatamente  a prob de como o evento ocorre na vida real.

**É A MESMA COISA QUE BINARYCROSS ENTROPY = TEORIA DA INFORMAÇÃO**

**EM suma: A log loss é minimizada (modelo perfeito) quando a prob prevista é igual a probabildiade real**

---

Quando usar: A log loss é imporante quando a probabilidade para classificar algo tem que ser bem calibrada.

quanto menor a log losss, melhor

````python
from sklearn.metrics import log_loss

print("P = {}\nY = {}".format(p_binary, y_binary))

log_loss(y_binary, p_binary)
# P = [0.49460165 0.2280831  0.25547392 0.39632991 0.3773151  0.99657423
#  0.4081972  0.77189399 0.76053669 0.31000935]
# Y = [0 0 0 1 1 1 0 1 1 0]
# 0.456820673923256

# Previsâo aleartória
p_random = np.ones(10) * 0.5
log_loss(y_binary, p_random)
# 0.6931471805599453

# Para uma previsâo binária, seu modelo deve estár abaixo de 0.69
````

### Curva ROC e AUC ROC

A curva ROC representa uma relação entre sensibilidade (RECALL - ) e especificidade (NÃO PRECISÃO) e é comumente usada para medir o desempenho de classificadores binários.

**Interpretação**
+ E quando mais curvado e distante da diagonal , melhor é o desempenho do seu modelo.
+ Quanto mais próximo a curva do seu modelo da diagonal pior será o desempenho do modelo.
**Parâmetros**

+ TPR (true Positive Rate - Taxa de Verdadeiro Positivo) também chamado de **Sensibilidade** [0,1]

+ FPR (false Positve Rate - Taxa de Falso Positivo) que é calculado como `1 - ` **Especificidade** [0,1]


<img src="../img/img-roc-auc-cap-04.png"  />

#### O que é ROC AUC, quando usar e como interpretar?

AUC (área debaixo da curva) ou AUC-ROC (Area Under the Receiver Operating Characteristics) é um valor numérico que resume a curva ROC.RC. Varia entre [0,1] quanto mais próximo de 1 melhor.

O interessante do AUC é que a métrica é invariante em escala, uma vez que trabalha com precisão das classificações ao invés de seus valores absolutos. Além disso, também mede a qualidade das previsões do modelo, independentemente do limiar de classificação.

AUC é o valor da integral da curva ROC. É um valor numérico entre \[0,1\].

É feito apartir do e TPR e FPR

Quanto maior o valor do AUC melhor será o modelo.

A seguir á alguns exemplos de gráficos ROC e valores AUC para entender a correlação entre eles

<div style="text-align: center;">
<img src="../img/img-roc-auc-cap-03.png" style="width: 60%" />
</div>
Exemplo de várias ROC

<div style="text-align: center;">
<img src="../img/img-roc-auc-cap-07.png"  />
</div>

**outra interpretação**

- Interpretar ROC-AUC: **Qual é a chance de um exemplo positivo ter um score (previsão) maior do que um negativo?**
- bom quando garantir que positivos sejam rankeados acima dos negativos é mais importante do que prever a probabilidade real do evento
  + Exemplo do Spam: Diferente da log loss,eu não me importa com a probabildiade (a certesa do modelo) em classificar se é span ou não (pois isso depende também no threshold). **Eu quero que o email que tenha mais cara de spam  mesmo seja devidamente classificado como span**

+ qual é a chance de um exemplo positivo ter um score (previsão) maior do que um negativo?
+ bom quando garantir que positivos sejam rankeados acima dos negativos é mais importante do que prever a probabilidade real do evento

Experimento

+ Suponha que tenha duas caixas, uma com só exemplos positivos e outra com apenas exemplo negativas.
+ Eu quero saber: vou tirar dessas caixas um exemplo positivo e um exemplo de negativo ver a probabildiades do meu modelo e devolver pra caixa (é uma coisa de probabilidade sem reposiçâo, possa pegar o mesmo mais de uma vez)
 - Olho a prob que meu modelo deu para esse exmeplo positivo
 - Olho a prob que meu modelo deu para o exemplo negativo

Se a prob do positivo é maior que negativo, entaô, conto +1.

A porcentagem de veze que o positov > negativo = AUC Score

AUC SCORE = **qual é a chance de um exemplo positivo ter uma prob maior que o do negativo**

É mais interressante quando eu quero saber que os positivos sejam mais identificáveis com certeza que os negativo (de certa forma um pouco relacionado com a Precision para os positivos).

````python
sum_over = 0
total = 100000

for i in range(total):

  caixa_de_positivos = p_binary[y_binary == 1] # caixa com só positivo
  caixa_de_negativos = p_binary[y_binary == 0] # caixa com só negativo

  positivo = np.random.choice(caixa_de_positivos, size=1, replace=False)
  negativo = np.random.choice(caixa_de_negativos, size=1, replace=False)

  if positivo > negativo:
    sum_over += 1

sum_over / total # AUC-ROC
````

### AUC da PRC - Area Under the Precision-Recall Curve
- É AVALIAR O ODELO INDEPNDENTE DO THRESHOLD
- acho mais estável e mais fácil de interpretar
- É uma média ponderada da curva de precision/recall
- **VOCÊ CONSEGUE AVALIAR INDEPENDENTE DO PONTO DE CORTE E ALÉM DISSO, VER O DESEMPENHO PARA VÁRIOS PONTOS DE CORTES DIFERENTES**
  + Assim, depois de usála, podemos escolher um ponto de corte bom

````python
from sklearn.metrics import average_precision_score
print("P = {}\nY = {}".format(p_binary, y_binary))

average_precision_score(y_binary, p_binary)
# P = [0.49460165 0.2280831  0.25547392 0.39632991 0.3773151  0.99657423
#  0.4081972  0.77189399 0.76053669 0.31000935]
# Y = [0 0 0 1 1 1 0 1 1 0]
# 0.8761904761904762
````


[tabela](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py)

[average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

---
---
---

## Engenharia de Features

### Quais técnicas utilizadas para tratamento de variáveis categóricas?

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

### Por que usar one-hot-encoding ? ‍

Se simplesmente codificamos variáveis categóricas com Label-Encoding, elas se tornam ordinais, o que pode levar a consequências indesejáveis. Nesse caso, os modelos lineares tratam uma feature com o valor 4 como duas vezes melhor do que uma feature de valor 2. A codificação one-hot-encoding permite representar uma variável categórica em um espaço vetorial numérico, o que garante que os vetores de cada categoria tenham distâncias iguais entre si. 

A abordagem de one-hot-encoding não é adequada para todas as situações, porque, usando-a com variáveis categóricas de alta cardinalidade (por exemplo, identificação do cliente), encontraremos problemas por aumentar demais a dimensionalidade.

### DS-001 - Como selecionar as features mais importantes de um DataSet?

Remove as features que estão correlacionadas (pois mostram uma mesma tendência)

É possível fazer isso com:
+ Forward Selection, Backward Selection, Stepwise Selection
+ Random Forest, Xgboost 
+ Lasso Regressão

Avalia as variáveis e selecione as melhores delas.


### O que fazer com dados corrompidos ou faltantes?

https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/

+ 1. Deletar as rows (se forem poucas estiverem corrompidas)
  - Prós:
    * A remoção completa dos dados com valores ausentes resulta em um modelo robusto e altamente preciso
    * A exclusão de uma linha ou coluna específica sem informações específicas é melhor, pois ela não tem um grande preso para predição
  - Contras:
    * Perda de informações e dados
    * Funciona mal se a porcentagem de valores ausentes for alta (digamos 30%), em comparação com o conjunto de dados inteiro

+ 2. Para variáveis numéricas substituir por média / mediana / moda
  - Prós:
    * Essa é uma abordagem melhor quando o tamanho dos dados é pequeno
    * Pode impedir a perda de dados, o que resulta na remoção de linhas e colunas
  - Contras:
    * Imputar as aproximações dos dados, variância e bias (o que é ruim)
    * Funciona mal em comparação com outro método de múltiplas imputações

+ 3. Atribuir a uma variável categórica valores exclusivos
  - Usar qualquer outro valor ou usar probabilidade para atribuir a cada valor único categórico, uma probabilidade e colocar nas rows
  - Prós:
    * Menos possibilidades com uma categoria extra, resultando em baixa variação após uma codificação quente - uma vez que é categórica
    * Nega a perda de dados adicionando uma categoria única
  - Contras:
    * Adiciona menos variação
    * Adiciona outro recurso ao modelo durante a codificação, o que pode resultar em baixo desempenho

+ 4. Prever valores faltantes
  - Pode-se usar regressão linear para predizer uma variável faltantes usando as outras que não tem valores faltantes
  - MELHORAR....
  - Prós:
    * Imputar a variável ausente é uma melhoria, desde que o viés da mesma seja menor que o viés da variável omitida
    * Gera estimativas imparciais dos parâmetros do modelo
  - Contras:
    * O viés também surge quando um conjunto de condicionamentos incompleto é usado para uma variável categórica
    * Considerado apenas como um *proxy* para os valores verdadeiros

+ 5. Usando algoritmos que suportam valores faltantes
  - Exemplo: KNN, Decision Tree e Random Forest
  - Prós:
    * Não requer a criação de um modelo preditivo para cada atributo com dados ausentes no conjunto de dados
    * A correlação dos dados é negligenciada
  - Contras:
    * É um processo muito demorado e pode ser crítico na mineração de dados onde grandes bancos de dados estão sendo extraídos
    * A escolha das funções de distância pode ser Euclidiana, Manhattan etc., o que não gera um resultado robusto

---
---
---

## Gradient boosting

### Como random forest é diferente de Gradient Boosting Machine (GBM)?

[https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/](https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)

[https://www.analyticsvidhya.com/blog/2020/02/4-boosting-algorithms-machine-learning/](https://www.analyticsvidhya.com/blog/2020/02/4-boosting-algorithms-machine-learning/)

A diferença fundamental é que a Random Forest usa a técnica de ensacamento para fazer previsões. 

O GBM usa técnicas de reforço para fazer previsões.

Na técnica de ensacamento, um conjunto de dados é dividido em n amostras usando amostragem aleatória. Em seguida, usando um único algoritmo de aprendizado, um modelo é construído em todas as amostras. Mais tarde, as previsões resultantes são **combinadas** usando votação ou média. O ensacamento é feito em paralelo. Ao aumentar, após a primeira rodada de previsões, o algoritmo pesa previsões mal classificadas mais altas, de modo que elas possam ser corrigidas na rodada seguinte. Esse processo seqüencial de atribuir pesos mais altos a previsões classificadas incorretamente continua até que um critério de parada seja alcançado.

A Random Forest melhora a precisão do modelo reduzindo a variação (principalmente). As árvores cultivadas não são correlacionadas para maximizar a diminuição da variação. Por outro lado, o GBM melhora a precisão, reduzindo o viés e a variação de um modelo.

---
---
---

## Reajuste de Hyper Parâmetros (melhorar modelo)

### Que estratégias de ajuste de hyper parâmetros você conhece?

Dissecar
[https://towardsdatascience.com/hyperparameter-tuning-explained-d0ebb2ba1d35](https://towardsdatascience.com/hyperparameter-tuning-explained-d0ebb2ba1d35)
[https://towardsdatascience.com/8-advanced-python-tricks-used-by-seasoned-programmers-757804975802](https://towardsdatascience.com/8-advanced-python-tricks-used-by-seasoned-programmers-757804975802)

Existem várias estratégias para o hiper-ajuste, mas eu argumentaria que as três mais populares atualmente são as seguintes:

A Grid Search (pesquisa em grade)  é uma abordagem exaustiva, de modo que, para cada hiper-parâmetro, o usuário precisa fornecer manualmente uma lista de valores para o algoritmo testar. Depois que esses valores são selecionados, a Grid Search avalia o algoritmo usando cada combinação de hiper-parâmetros e retorna a combinação que fornece o resultado ideal (ou seja, MAE mais baixo). Como a grid search avalia o algoritmo fornecido usando todas as combinações, é fácil ver que isso pode ser bastante computacional e pode levar a resultados abaixo do ideal, uma vez que o usuário precisa especificar valores específicos para esses hiper-parâmetros, o que é propenso a erros e requer conhecimento de domínio.

...


---
---
---

## Feature Selection

[link mosntruoso](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)

+ Feature Selection: Select a subset of input features from the dataset.
  - Unsupervised: Do not use the target variable (e.g. remove redundant variables).
    *Correlation
  - Supervised: Use the target variable (e.g. remove irrelevant variables).
    * Wrapper: Search for well-performing subsets of features.
      + RFE
    * Filter: Select subsets of features based on their relationship with the target.
      + Statistical Methods
      + Feature Importance Methods
    * Intrinsic: Algorithms that perform automatic feature selection during training.
      + Decision Trees
+ Dimensionality Reduction: Project input data into a lower-dimensional feature space.


<div style="text-align: center;">
<img src="../img/features-selection-01.png"  />
<img src="../img/features-selection-02.png"  />
<img src="../img/features-selection-03.png"  />
</div>

+ Numerical Variables
  - Integer Variables.
  - Floating Point Variables.
+ Categorical Variables.
  - Boolean Variables (dichotomous).
  - Ordinal Variables.
  - Nominal Variables.

Numerical Output: Regression predictive modeling problem.
Categorical Output: Classification predictive modeling problem.


How to Choose Feature Selection Methods For Machine Learning

### Numerical Input, Numerical Output

This is a regression predictive modeling problem with numerical input variables.

The most common techniques are to use a correlation coefficient, such as Pearson’s for a linear correlation, or rank-based methods for a nonlinear correlation.

- Pearson’s correlation coefficient (linear).
- Spearman’s rank coefficient (nonlinear)

### Numerical Input, Categorical Output

This is a classification predictive modeling problem with numerical input variables.

This might be the most common example of a classification problem,

Again, the most common techniques are correlation based, although in this case, they must take the categorical target into account.

- ANOVA correlation coefficient (linear).
- Kendall’s rank coefficient (nonlinear).

Kendall does assume that the categorical variable is ordinal.

### Categorical Input, Numerical Output

This is a regression predictive modeling problem with categorical input variables.

This is a strange example of a regression problem (e.g. you would not encounter it often).

Nevertheless, you can use the same “*Numerical Input, Categorical Output*” methods (described above), but in reverse.

### Categorical Input, Categorical Output

This is a classification predictive modeling problem with categorical input variables.

The most common correlation measure for categorical data is the [chi-squared test](https://machinelearningmastery.com/chi-squared-test-for-machine-learning/). You can also use mutual information (information gain) from the field of information theory.

- Chi-Squared test (contingency tables).
- Mutual Information.

In fact, mutual information is a powerful method that may prove useful for both categorical and numerical data, e.g. it is agnostic to the data types.

## Tips and Tricks for Feature Selection

This section provides some additional considerations when using filter-based feature selection.

### Correlation Statistics

The scikit-learn library provides an implementation of most of the useful statistical measures.

For example:

- Pearson’s Correlation Coefficient: [f_regression()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
- ANOVA: [f_classif()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
- Chi-Squared: [chi2()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)
- Mutual Information: [mutual_info_classif()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) and [mutual_info_regression()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)

Also, the SciPy library provides an implementation of many more statistics, such as Kendall’s tau ([kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)) and Spearman’s rank correlation ([spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)).

### Selection Method

The scikit-learn library also provides many different filtering methods once statistics have been calculated for each input variable with the target.

Two of the more popular methods include:

- Select the top k variables: [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
- Select the top percentile variables: [SelectPercentile](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html)

I often use *SelectKBest* myself.

### Transform Variables

Consider transforming the variables in order to access different statistical methods.

For example, you can transform a categorical variable to ordinal, even if it is not, and see if any interesting results come out.

You can also make a numerical variable discrete (e.g. bins); try categorical-based measures.

Some statistical measures assume properties of the variables, such as Pearson’s that assumes a Gaussian probability distribution to the observations and a linear relationship. You can transform the data to meet the expectations of the test and try the test regardless of the expectations and compare results.

### What Is the Best Method?

There is no best feature selection method.

Just like there is no best set of input variables or best machine learning algorithm. At least not universally.

Instead, you must discover what works best for your specific problem using careful systematic experimentation.

Try a range of different models fit on different subsets of features chosen via different statistical measures and discover what works best for your specific problem.

### Quando fazer

1. ** Você tem conhecimento de domínio? ** Se sim, construa um conjunto melhor de recursos ad hoc ””
2. ** Seus recursos são proporcionais? ** Se não, considere normalizá-los.
3. ** Você suspeita de interdependência de recursos? ** Em caso afirmativo, expanda seu conjunto de recursos construindo recursos conjuntivos ou produtos de recursos, tanto quanto os recursos do seu computador permitirem.
4. ** Você precisa remover as variáveis ​​de entrada (por exemplo, por razões de entendimento de custo, velocidade ou dados)? ** Se não, construa recursos disjuntivos ou somas ponderadas de recursos
5. ** Você precisa avaliar os recursos individualmente (por exemplo, para entender sua influência no sistema ou porque seu número é tão grande que você precisa fazer uma primeira filtragem)? ** Se sim, use um método de classificação variável; caso contrário, faça-o assim mesmo para obter resultados de linha de base.
6. ** Você precisa de um preditor? ** Se não, pare
7. ** Você suspeita que seus dados estejam "sujos" (possui alguns padrões de entrada sem sentido e / ou saídas ruidosas ou rótulos de classe incorretos)? ** Se sim, detecte os exemplos outlier usando as principais variáveis ​​de classificação obtidas na etapa 5 como representação; verifique e / ou descarte-os.
8. ** Você sabe o que tentar primeiro? ** Se não, use um preditor linear. Use um método de seleção direta com o método "probe" como critério de parada ou use o método incorporado de norma 0 para comparação, seguindo a classificação da etapa 5, construa uma sequência de preditores da mesma natureza usando subconjuntos crescentes de recursos. Você pode igualar ou melhorar o desempenho com um subconjunto menor? Se sim, tente um preditor não linear com esse subconjunto.
9. ** Você tem novas idéias, tempo, recursos computacionais e exemplos suficientes? ** Se sim, compare vários métodos de seleção de recursos, incluindo sua nova ideia, coeficientes de correlação, seleção reversa e métodos incorporados. Use preditores lineares e não lineares. Selecione a melhor abordagem com a seleção de modelos
10. ** Deseja uma solução estável (para melhorar o desempenho e / ou a compreensão)? ** Se sim, subamostra os dados e refaça a análise para várias "instruções de inicialização".

---
---
---

<!-- 

## Dimensionality reduction

### What is the curse of dimensionality? Why do we care about it? ‍⭐️

Data in only one dimension is relatively tightly packed. Adding a dimension stretches the points across that dimension, pushing them further apart. Additional dimensions spread the data even further making high dimensional data extremely sparse. We care about it, because it is difficult to use machine learning in sparse spaces.

### Do you know any dimensionality reduction techniques? ‍⭐️

Singular Value Decomposition (SVD)
Principal Component Analysis (PCA)
Linear Discriminant Analysis (LDA)
T-distributed Stochastic Neighbor Embedding (t-SNE)
Autoencoders
Fourier and Wavelet Transforms


### What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️

Singular Value Decomposition (SVD) is a general matrix decomposition method that factors a matrix X into three matrices L (left singular values), Σ (diagonal matrix) and R^T (right singular values).
For machine learning, Principal Component Analysis (PCA) is typically used. It is a special type of SVD where the singular values correspond to the eigenvectors and the values of the diagonal matrix are the squares of the eigenvalues. We use these features as they are statistically descriptive.
Having calculated the eigenvectors and eigenvalues, we can use the Kaiser-Guttman criterion, a scree plot or the proportion of explained variance to determine the principal components (i.e. the final dimensionality) that are useful for dimensionality reduction.

## O que é PCA?

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

Significa: "Análise de Componetnes principais"

=> Para problemas linearmente separáveis

=> Deve-se ser feito asobre dados numéricos

**Diferenciar Seleçâo de Extração de características**

• Seleção de características x Extração de características
+ Seleção: Indicar os atributos mais importantes
+ Extração: AO fazer uma análise, criar novos atributos. É como unir atributos.
  - Encontrar relacionamentos entre os atributos para combinar eles e reduzir a dimensionalidade

Lembre-se, o PCA nâo é escolher os melhore "n" atributos, e sim reduzir para "n" atributos

**Características do PCA**

• PCA: Identifica a correlação entre variáveis, e caso haja uma forte
correlação é possível reduzir a dimensionalidade.
  - Exemplo: Se vocÊ têm duas variáveis com forte correlação vocÊ pde então unilas e asism reduzir em 1 a quantidade de features.

**Funcionamento**

• Um dos principais algoritmos de aprendizagem de máquina não
supervisionada (não há certo/errado apriori)
• Das m variáveis independentes, PCA extrai p <= m novas variáveis
independentes que explica melhor a variação na base de dados, sem
considerar a variável dependente
• O usuário pode escolher o número de p

COMPONETSE PRICINCPAIS SÂO OS COMEPONETNE DE features concatenados.

**Se com PCA fiacr pior?**

Abaixou um pouco, mas, você deve avaliar o `trade_off` entre a precisao e a velocidade.

Exemplo, será que mesmo reduzindo 1\% poderia ser melhor usar o PCA pois teria menos dados para classificar e assim ter menos custo computacional?

````python

pca.explained_variance_ratio_ Revela o quanto os componentes definem os dados, a importância deles.

Inicialmente, deve-se colocar PCA(n_components = None) para descobrir a variance_ratio para cada feature. Depois disos, escolher um número menor que len(features) cuja soma seja alta.

Um exemplo, prático. Calcular o PCA para 30 feautures e descobrir que, 3 features tem a soma de variance_ratio de 0.80. Ou seja, somente 3 variáveis já seriam necessário para contextualizar sua base com 80% de veracidade.

componentes = pca.explained_variance_ratio_
componentes
executed in 17ms, finished 16:07:54 2020-01-07
array([0.151561  , 0.10109701, 0.08980379, 0.08076277, 0.07627678,
       0.07357646])
componentes.sum()
executed in 31ms, finished 16:42:58 2020-01-07
0.5730778058904467

````
## LDA

**Questions LDA - English ML-AZ**
LDA Intuition
Could you please explain in a more simpler way the difference between PCA and LDA?
A simple way of viewing the difference between PCA and LDA is that PCA treats the entire data set as a whole while LDA attempts to model the differences between classes within the data. Also, PCA extracts some components that explain the most the variance, while LDA extracts some components that maximize class separability.

Feature Selection or Feature Extraction?
You would rather choose feature selection if you want to keep all the interpretation of your problem, your dataset and your model results. But if you don’t care about the interpretation and only car about getting accurate predictions, then you can try both, separately or together, and compare the performance results. So yes feature selection and feature extraction can be applied simultaneously in a given problem.

Can we use LDA for Regression?
LDA is Linear Discriminant Analysis. It is a generalization of Fisher’s linear discriminant, a method used in statistics, pattern recognition and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification. However, for regression, we have to use ANOVA, a variation of LDA. LDA is also closely related to principal component analysis (PCA) and factor analysis in that they both look for linear combinations of variables which best explain the data. LDA explicitly attempts to model the difference between the classes of data. PCA on the other hand does not take into account any difference in class, and factor analysis builds the feature combinations based on differences rather than similarities. Discriminant analysis is also different from factor analysis in that it is not an interdependence technique: a distinction between independent variables and dependent variables (also called criterion variables) must be made. LDA works when the measurements made on independent variables for each observation are continuous quantities. When dealing with categorical independent variables, the equivalent technique is discriminant correspondence analysis.

LDA in Python
Which independent variables are found after applying LDA?
The two independent variables that you see, indexed by 0 and 1, are new independent variables that are not among your 12 original independent variables. These are totally new independent variables that were extracted through LDA, and that’s why we call LDA Feature Extraction, as opposed to Feature Selection where you keep some of your original independent variables.

How to decide the LDA n_component parameter in order to find the most accurate result?
You can run:

LDA(n\_components = None
and it should give you automatically the ideal n_components.

How can I get the two Linear Discriminants LD1 and LD2 in Python?
You can get them by running the following line of code:

lda.scalings_

---

**LDA (Linear Discriminant Analysis)**

Possui mesma função do PCA mas envolve a classe dos dados, ou seja, é um algortimo supervisionado.

=> Para problemas linearmente separáveis

• Além de encontrar os componentes principais, LDA também encontra os eixos que maximizam a separação entre múltiplas classes

• É um algoritmo supervisionado por causa da relação que tem com as classes

• Das m variáveis independentes, LDA extrai p <= m novas variáveis independentes que mais separam as classes da variável dependente

-->

## Add

+ + Tratar outilers no y de regresssâo: winsorizar
+ Se precizar de GPU para coisas extermamente seriesas, pode-se pensar em alugar a AWS que tem GPU pra essas coisas.
+ É sempre usar métricas de acordo com o negócio que está resolvendo: 
  - Em geral selecione uma primaria de acordo com o problema
   - Dpeois escolheas outra para ter mais ângulos de observação

   => Uma ideia interressante seria fazer um modelo focado em recall primiero e depois outro apra precision

==> recall x precision: se detectar positivos/negativos for mais importante que acertar positivos/negativos, entâo recall é m ais imporatnte

==> Se nâo há ponto de corte e dados desbalanceados: AUC e PRC 

==> Como saber se ocorreu ovefiting: 
+ Em geral é quando score de testes está diferente do score de treinamento

+ Quando fazer feature selection:
  - Quando houver muitas features com um dataset pequeno
  - Usa-se o valor p (pearson) para encontrar a correlação entre x e y
  - As vezes, em vez de fazer feature selection, para regressão, pode ser necesśario fazer Lasso//Ridge

  HUGE GLOSSARY

  https://ml-cheatsheet.readthedocs.io/en/latest/index.html
  https://peltarion.com/knowledge-center/documentation/glossary
  http://deeplearningbook.com.br/