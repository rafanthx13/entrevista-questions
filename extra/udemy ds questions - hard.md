# Level Hard

## NLP

### p07

Em processamento de linguagem natural, é o processo de adicionar tags a cada token, de acordo com o elemento encontrado, como substantivo, adjetivo etc.

Parts-of-Speech Tagging

Explicação
Este processo é conhecido como Parts-of-Speech Tagging, ou simplesmente POS

---
---
---

## Séries Temporais

### p11

Qual opção melhor descreve o modelo Arma(1,1) ou Arima(1,0,1)

Autoregressivo de primeira ordem e média móvel de primeira ordem

Explicação
Temos aqui os parâmetros p e q, de ordem de média móvel e ordem auto-regressiva, sendo grau de diferenciação igual a zero

### p13

Observe o diagrama de auto correlação da imagem abaixo e marque a opção que melhor explica a série temporal:

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_14-57-19-52b62190185b39df588f1d09d4b432f3.png

Uma série temporal com tendência

Explicação
O padrão apresentado pelo diagrama ACF é tipico de uma série temporal com tendência

### p22

Das características de uma série temporal descrita como passeio aleatório, marque a opção falsa:

Pode ser modelado matematicamente

Explicação
Um passeio aleatório não pode ser explicado matematicamente, por ser um evento aleatório, sem padrão

### p23

Pensando-se em processo de forecast de séries temporais usando Arima, com as seguintes etapas, que podem estar fora de ordem:

1) Análise de residuais

2) Análise Exploratória,

3) Estabilização,

4) Previsão

5) Criação do modelo

Marque a opção que apresenta a ordem correta em que estes processos devem ser executados:

>  2,3,4,1,4

Explicação
O primeiro passo é conhecer os dados com análise exploratória. Em seguida, se necessário, executa-se algum processo de estabilização ou transformação da série temporal. Então são criados os modelos e em seguida analise-se os resíduos para verificar a qualidade do modelo. Só então fazemos a previsão

### p25

Observe o diagrama de auto correlação da imagem abaixo e marque a opção que melhor explica a série temporal:

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_14-58-22-7b430fb86bbe768df60066840b37048a.png

Uma série temporal do tipo  random wak

Explicação
Não se observa autocorrelação na série temporal, o que indica random walk

### p27

Em séries temporais, quais características devemos esperar de resíduos de um bom modelo? (marque todas que se aplicam)


Média próxima de zero; Normmalmente distrivuidos, Varianca constante; Não estarem autocorrelacionados

Explicação
Todas são características que esperamos encontrar em um bom modelo de séries temporais

### p31

Você está avaliando resíduos de uma série temporal e executa um teste de Ljung-Box, com valor de p=0,35. O que podemos concluir com este teste?

<p>Há indícios de que os dados não estão autocorrelacionados</p>

Explicação
Um valor de p > 0,05 traz indícios de que não há autocorrelação

### p45

Marque as opção verdadeiras com relação a criação de modelos com dados com componentes sazonais com Arima: (marque todas que se aplicam)

+ Pode-se remover o elemento sazonal e criar um modelo com arima padrão
+ O modelo inclui os elementos P,D e Q para dados sazonais

Explicação
Arima permite criação de modelos sazonais com os elementos de ordem de média móvel, grau de diferenciação e ordem da parte auto-regressiva específicos para este componente, ou podemos remover o elemento sazonal e usar os elementos p, d e q padrão do modelo

### p73

Com dados de uma série temporal, executou-se o teste de Dickey-Fuller, chegando-se a um valor de p = 0.455. Qual alternativa tem a melhor resposta para a conclusão do resultado deste teste?

Não conseguimos rejeitar a Hipótese Nula, de não estacionariedade

Explicação
No teste de Dickey-Fuller, a hipótese nula é de não estacionáridade. Com p = 0.455, rejeitamos a Hipótese Nula, de não estacionariedade

### p95

Qual opção abaixo melhor descreve o modelo ARIMA( (0,1,0)?

Passeio aleartório

Explicação
O modelo Arima(0,1,0) é um passeio aleatório

### p98

Na imagem abaixo, foram utilizadas duas técnicas de forecast de séries temporais para um mesmo conjunto de dados. Das opções abaixo, qual explicaria melhor a diferença que podemos notar na previsão das duas séries?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_15-00-36-dbf42476df2e089aa6b52bb5e8139f6f.png

> Ambas as previsões podem ter utilizado algoritmos de suavização exponencial, mas na segunda previsão, foi claramente aplicado um processo de suavização da previsão

Explicação
Os modelos arima apresentados equivalem a random walk e suavização exponencial linear, que poderia ser o caso apenas do primeiro gráfico. Portanto a resposta correta é a 4

---
---
---

## Regressâo e Correlação Linear

### p28

Observe os dados da tabela abaixo. Para um problema de regressão linear, calcule a interceptação da reta, sabendo que a inclinação da reta é igual a 13,02:

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-30_17-08-11-0fa6ca40634292896a4af30963f4c9b8.PNG

> 179,05

Explicação
A interceptação é a média de y menos o produto da inclinação pela média de x

### p36

Observe os dados da tabela abaixo. Para um problema de regressão linear, preveja Y para X = 14, sendo que a inclinação da reta é igual a 13,02 e a interceptação da reta é igual a 179,05:

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-30_17-08-11-0fa6ca40634292896a4af30963f4c9b8.PNG

> 361,33

Explicação
Se obtêm a previsão somando-se a interceptação mais o produto da inclinação pela valor da variável independente

### p50

Observe os dados da tabela abaixo. Para um problema de regressão linear, calcule a inclinação da reta, sabendo que o coeficiente de correlação é igual a 0,97.

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-30_17-08-11-0fa6ca40634292896a4af30963f4c9b8.PNG

> 13,02

Explicação
A inclinação é calculada multiplicando a correlação pelo desvio padrão de X dividido pelo desvio padrão de Y

### p57

Qual observação é falsa com relação a extrapolação de modelos de regressão linear?

Deve-se observar caso a caso, a extrapolação deve fazer sentido para o negócio que esta sendo modelado

Explicação
Quando falamos de regressão linear, temos um linha que cresce ao infinito. Para a maioria dos modelos de negócio haverá um ponto em que não fará mais sentido prever. Por exemplo, prever quantos casacos serão vendidos quando a temperatura for igual a -60 graus Celsius

### p69

Calcule a correlação de Person das variáveis da tabela abaixo.

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-30_17-08-11-0fa6ca40634292896a4af30963f4c9b8.PNG

> 0.97

Explicação
Deve-se calcular a covariação de x e y e dividir pela raiz quadrada da variância de x multiplicado pela variância de y

---
---
---

## Medidas de variabilidade e centralidade

### p51

Dada a variável n cujos valores são 234, 56, 345, 754, 4, 1345, calcule a sua mediana:

> 289,5

Explicação
Ordenamos os valores: 4, 56, 234, 345, 754, 1345. Como a quantidade é par, calculamos a média dos dois valores centrais: 234 + 345 / 2 = 289,5

### p65

Qual é a amplitude do conjunto de dados abaixo:  {23,45,67,8,23,87}

> 79

Explicação
Amplitude é o maior valor menos o menor valor, 87 - 8 = 79

---
---
---

## Redes Neurais Artificiais e Deep Learing

### p01

O que acontece se submetermos a uma rede neural artificial do tipo Perceptron, dados não separáveis linearmente?

Nunca vai convergir

Explicação
A rede neural nunca vai convergir, devido a sua natureza simples. A solução seria, por exemplo, definir um número máximo de epochs

### p08

Existe uma regra geral para definição de arquitetura de redes neurais artificiais? Marque a opção correta:

Quantidade de atributos mais quantidade de classes dividido por dois

Explicação
A regra geral é (a+c)/2, ou atributos mais classes dividido por dois

### p21

Representação gráfica cujo objetivo é comparar verdadeiros positivos com falsos positivos em vários intervalos.

Curva COR (ROC CUrve)

Explicação
O Gráfico de Curva ROC tem como objetivo comparar verdadeiros positivos com falsos positivos

### p35

Como um modelo de Deep Learning generativo pode ser utilizado de forma discriminativa?

Classificando o objeto pela distribuição da classe que apresentar a maior probabilidade

Explicação
Como o modelo generativo gera uma distribuição de probabilidade da classe, basta olhar qual tem a maior probabilidade para rotular o objeto

### p37

Existem algumas variações da técnica de Gradient Descent com relação ao cálculo do gradiente. Qual opção melhor define a técnica stochastic?

Usa apenas um exemplo de treino para o cálculo

Explicação
Na técnica stochastic temos apenas uma atualização do gradiente com dados de treino

### p94

Em um modelo de redes neurais artificiais, se definiu um valor baixo para gama (taxa de aprendizado). Qual o comportamento esperado da rede com este parâmetro?

vai convergir muito lenetamnete

Explicação
Gama define a taxa de aprendizado. Se ele é muito alto a rede fica lenta pois tem que fazer grades ajustes nos pesos, porém se é muito baixo, as correções são muito pequenas e também pode demorar para convergir.

### p96

Qual opção apresenta a melhor distinção entre modelos discriminativos e generativos?


> O generativos aprende a distinguir entre diferentes entradas, classificando como sim ou não, por exemplo. Modelos discriminativos envolvem uma distribuição de probabilidade sobre a classificação

Explicação
O generativo na verdade apresenta a probabilidade da imagem ser ou não determinado objeto, enquanto o discriminativo vai classificar com ser ou não aquele objeto

### p97

Sobre Mapas de Kohonen, marque a opção falsa:

FALSE: É uma tarefa supervisioanda

ENTAO AS VERDADEIRAS SÃO:
+ Baseada e0 c60*et5t5ve 3earn5ng
+ eh do tipo auto-organizavel
+ eh uma neral net

Explicação
Mapas de Kohonen são utilizados para tarefas não supervisionadas, especificamente, agrupamentos

---
---
---

## Estatística

### p03

Quantas modas existem neste conjunto de dados:

{1,2,2,5,5,5,8,10,10,12,18,21,21,29,29,29}

> 3

Explicação
Existem duas modas, 5,5,5 e 29,29,29

### p16

Marque todas as alternativas verdadeiras com relação a distribuição de Poisson:

+ Mede a prob. da ocorrencia de eventos em intervalo de tempo
+ Os eventos devem ser independentes

Explicação
A distribuição de Poisson é uma distribuição de probabilidade que mede a ocorrência de eventos em intervalos de tempo

### p66

Qual opção melhor descreve o método de correlação?

Mede a relação entre duas variáveis

Explicação
A correlação mede a força da relação matemática entre duas variáveis

### p74

Marque a opção que não faz parte dos objetivos da analise de dados exploratória:

> Prever o valor de açôes

> VERDADEIRO: Detectar variaçôes; Buscar anomalias; encontradr tendencias

Explicação
A EDA busca um compreensão mais ampla dos dados e do fenômeno que os criaram, mas não tem por objetivo fazer previsões

### p79

De acordo com a imagem abaixo, qual é a fórmula correta para transformar o valor de uma variável normalmente distribuída na distribuição normal padrão (Z)? (marque de acordo com a numeração)

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_13-41-35-f4571a2b7a29ce36b26dd4de233a3f95.PNG

> 3

Explicação
A fórmula correta é a variável subtraindo a média dividido pelo desvio padrão

### p82

Em média, uma concessionária vende 2 carros por dia. Qual a probabilidade de, em um determinado dia, vender 3 carros?

> 0.1804

Explicação
Utilizando Distribuição de Poisson, sendo a constante e = 2,71828, temos 2,71828^2 * 2^3, então 0,1353355 * 1,3333 = 0,1804

---
---
---

##Inteligência artificial

### p06 

Em algoritmos genéticos, qual é a melhor definição para o processo descrito como elitismo?

Processo de transmitir um cromossomo para a próxima geração sem qualquer alteração

Explicação
O elitismo busca manter as características mais bem adaptadas de uma geração para outra

### p09

Qual é a principal diferença entre uma tarefa de machine learning como classificação, e uma tarefa de reinforcement learning?

Reinforcement learning aprende pela interação com o ambiente, machine learning com dados históricos

Explicação
A principal diferença é que machine learning aprende com dados, enquanto Reinforcement Learning aprende pela interação com o ambiente

### p12

Observe a imagem abaixo. Que tipo de sistema especialista foi implementado?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-29_14-33-23-223b0368d05211311a4792104c88ecd2.png

Sistema especialista baseado em lógica difusa (fuzzy)

Explicação
O sistema de lógica difusa apresenta lógicas parciais, que não pertencem necessariamente a uma úncia categoria, o que fica claramente representado no imagem

### p19

Existem algoritmos de busca que procuram uma solução global, outros, uma solução local (em um subconjunto de soluções possíveis). Qual a principal razão de não utilizarmos sempre procedimentos de busca global?

Quando não é possível buscar globalmente

Explicação
As vezes não é possível buscar globalmente, pode levar muito tempo, pode não ser possível processar (capacidade computacional) entre outros

### p33

Qual a principal desvantagem do algoritmo de busca Hill Climbing?

Pode ficar "preso" em uma soluçâo mínima local

Explicação
Hill Climbing pode por acaso chegar ao valor ótimo global, então sua principal desvantagem é ficar preso em uma solução mínima local

### p34

Sobre a técnica de otimização conhecida como Algoritmos Genéticos, marque a opção com a melhor definição:

Utiliza processos evolutivos, com novas gerações mais adaptadas

Explicação
Algoritmos genéticos são inspirados no processo de evolução natural das espécies

### p38

Sobre o algoritmo de otimização Simulated Annealing, marque a opção correta:

Mantém uma variável temperatura. Onde a temperatura é alta, explora soluções que aparentemente não otimização a função objeto

Explicação
Esta técnica se inspira no processo de aquecer e resfriar metal para alterar sua estrutura

### p39

Quando um algoritmo genético propõe uma nova solução para um problema, de que forma esta solução é proposta?

Uma nova geraçâo de cromossomos

Explicação
A solução é sempre proposta por vários novos cromossomos, que em teoria estão melhor adaptados do que a geração anterior

### p67

Sobre a técnica de otimização Tabu Search, marque a opção correta:

Gera e mantém uma lista de locais proibidos

Explicação
Tabu search mantém uma lista de locais proibidos (Tabu) em memória. Proibidos ou por já terem sido visitados ou por não otimizarem a função objetivo

### p81

Uma grupo hospitalar deseja criar um sistema de diagnóstico de doenças, para auxiliar médicos menos experientes. Inicialmente se pensou em um sistema baseado em machine learning, porém, não são coletados dados sobre as doenças especificas do qual se quer criar o sistema. Qual seria uma solução viável e mais rápida para desenvolver um sistema para atender o Hospital?

 Criar um sistema especialista

Explicação
Um sistema especialista pode ser alimentado com informações oriundas de médicos mais experientes, de forma que pode ser consulado por outros médicos. Muito embora se possa optar por criar um sistema para coletar dados, esta solução exigiria muito mais tempo

### p83

Sendo a imagem abaixo a simulação de um processo de aprendizado por reforço cujo objetivo é a criança aprender o caminho até chegar ao brinquedo, qual seria o comportamento do sistema se a criança tentasse passar do estado S3 para o estado S1?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_01-05-09-6cc9bcf2b9f8ab4a9c56fb472dc68aa8.PNG

> O sistema não seria compensado, pois é um movimento impossível

Explicação
No aprendizado por reforço, uma ação desconhece as consequências e recompensas de outras ações. Portanto, neste movimento o sistema não seria recompensado, ou teria uma recompensa ruim em termos de peso (-1 por exemplo)

---
---
---

## Machine Learning

### p02

Random Forest induz muitas árvores de decisão, ocorrendo a criação de partições usando boosting e com sub conjuntos de atributos, são induzidas diferentes árvores, que podem classificar uma mesma instância para diferentes classes. Neste caso, como é resolvido o problema da definição da classe da instância?

Por um procesos de votação

Explicação
Um simples processo de votação define a classe da instância.

### p04

Qual fórmula listado abaixo é a correta para o cálculo dos Positivos Falsos em um processo de treino de Machine Learning?

FP/(VN+FP)

Explicação
Positivos Falsos são calculados com Falsos Positivos divididos pela soma dos Verdadeiros Positivos mais Falsos Negativos

### p15

Hamming Lost é uma métrica de avaliação de performance em classificação multilabel. Na imagem abaixo, os itens pintados em vermelho são erros, todos os demais são acertos. Calcule o índice de Hamming Loss para este caso.

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_18-39-53-a5626afb34faa8b741cd135eeece05c8.PNG

> 0,1

Explicação
Temos 4 erros de um total de 40 posições, 4/40 = 0,1

### p20

Em machine learning, qual é a principal aplicação do procedimento conhecido como PCA (Principal component analysis)?

É uma técninac de reduçâo de dimensionalidade

Explicação
PCA é uma técnica de redução de dimensionalidade, ou seja, redução de características de uma relação

### p32

Das alternativas abaixo, marque as que são consideradas condições de parada na indução de uma árvore de decisão:

+ A pureza não aumenta
+ Chega-se a uma classe pura
+ O número de instâncias naquele ponto chegou a um valor mínimo definido

Explicação
Todas as opções são condições de parada, porém a profundidade da árvore não deve ser utilizada neste aspecto

### p40

Em machine learning, qual o principal objetivo do processo conhecido como regularização?

Reduzir super ajuste

Explicação
A regularização aciona parâmetros ao modelo de forma a minimizar super ajuste

### p41

São exemplos de classificadores conhecidos como métodos de grupos. Marque todas as opções que se aplicam.

> Adaboost; Boosting, Baggin, Random Forest

Explicação
Todas as opções são classificadores de grupos válidos

### p42

Na imagem abaixo, temos um processo de aprendizado baseado em instância. No circulo menor, temos k=4. No círculo maior, temos k=7. Qual será a classe para estas duas configurações, respectivamente?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_17-55-31-87f3918a9d09c53e5b7a11c9dc278a5b.PNG

> Quadrado e círculo

Explicação
Está se usando KNN, que define a classe por um processo de votação. Na primeira parametrização temos claramente mais quadrados (3 a 1). Na segunda parametrização, temos mais círculos (4 x3). Então a resposta é, respectivamente, quadrado e circulo

### p43

Com relação a Featuring Scaling, marque a alternativa que possui a melhor definição do conceito:

Criam-se escalas da influência dos atributos na classificação, de forma a melhor a performance de um modelo

Explicação
Featuring Scaling cria um ranking de como os atributos influenciam a classificação, de forma que podemos selecionar os que tem melhor influência e remover os que influenciam de forma negativa

### p44

Marque a opção que é um popular algoritmo de classificação do tipo árvores de decisão:

ID3

Explicação
A única opção que é um classificador de árvore de decisão é a ID3.

### p46

Qual opção abaixo é a melhor definição para uma função de custo?

É uma métrica do custo de prever X quando na verdade a previsão correta seria Y

Explicação
Uma função de custo avalia o custo de previsões

### p47

Qual fórmula listado abaixo é a correta para o cálculo da Lembrança (ou positivos verdadeiros) em um processo de treino de Machine Learning?

VP/(VP+FN)

Explicação
A lembrança é calculada com os verdadeiros positivos divididos pelos verdadeiros positivos mais os falsos negativos

### p49

O gráfico da imagem abaixo mostra a métrica conhecida como WSS (Within sum of squares) em gráfico conhecido como Elbone Curve. Qual a função primária deste tipo de gráfico?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_18-48-15-65e952bb571e6c588e995cbeef9ce230.PNG

Determinar o número de grupos para uma tarefa de agrupamento

Explicação
Em algoritmos de agrupamentos como K-means, podemos usar um gráfico Elbow Curve para avaliar o número de grupos para serem criados pelo agrupador

### p52

Dos itens abaixo, qual é a vantagem mais importante dos classificadores baseados em regras se comparados aos classificadores baseados em árvores de decisão?

Regras permitem analisar um conjunto mais amplo de decisões a serem tomadas, pois não estão limitadas a estrutura de uma árvore

Explicação
Muito embora a opção 4 (4: Regras de decisão podem incorporar a estrutura de uma árvore de decisão) não esteja errada, existem estruturas de decisão que não podem ser representadas por árvores, isso faz com que esta opção seja a mais importante, como diz o enunciado da questão

### p54

As etapas para produção de agrupamentos utilizando K-means abaixo, estão foram de ordem:

+ A - Agrupa os dados a um centroide
+ B - Recalcula cada média no centro dos pontos
+ C - Inicializa os centroides
+ D - Repete até os grupos não mudarem

A ordem correta das etapa é:

> C,A,B,D

Explicação
Primeiro é preciso inicializar os centroides, depois agrupar os dados próximos a cada centroide, recalcular cada média e repetir o processo

### p56

Os dados abaixo mostram na primeira coluna a previsão dos dados de treino, e na segunda coluna, a previsão obtida nos testes com o modelo. Baseado nestes dados, calcula a métrica MAE (Mean Absolute Erros ) e marque a opção com o resultado correto:

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_18-41-33-d05e77a140d02b8cfae2a5954588855f.PNG

> 0.15857

Explicação
Divide-se a diferença absoluta dos valores pelo número de instâncias (7)

### p59

Em clusters hierárquicos, qual opção abaixo apresenta a definição correta para Complete Linkage?

É a distância entre os dois pontos mais distantes de dois diferentes grupos

Explicação
O conceito correto é a distância entre os dois pontos mais distantes de dois diferentes grupos

### p60

Dadas as tabelas de probabilidade condicionais da imagem abaixo, submete-se ao modelo a seguinte instância:

outlook=sunny, temperature=hot, humidity=high, windy=FALSE.

Considerando que as classe são yes e no, qual será a classificação proposta pelo modelo para a instância e qual será o valor da probabilidade posterior calculada?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-30_17-24-17-74c9b839387c2f326b6a8af17eb5ef74.PNG

> Ni = 0,00538588

Explicação
Outlook, P(outlook|Play), Temperature P(temperature|outlook, play), Humidity P(Humidity|temperature,Play), Windy P(Windy|outlook,Play) Então: P(yes) = 0,633 * 0,238 * 0,143 * 0,5 * 0,5 = 0,00538588 P(no) = 0,367 * 0,538 * 0,556 * 0,833 * 0,5= 0,045723

### p61

Dado o modelo abaixo, criado por um classificador Naive Bayes, submetemos ao modelo a seguinte instância:

Outlook=sunny, temperature=mild, humidity=high, windy=False.

Baseado no modelo, como o classificador deverá classificar esta instância e com qual probabilidade posterior?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-30_17-28-38-dc5ad70ebc7d6950bf969f34a490962e.PNG

> NO = 0,02688

Explicação
Probabilidade YES: P(yes) * P(sunny|yes) * P(mild|yes) * P(High|Yes) * P(FALSE|yes). 0,64 * 0,22 * 0,44 * 0,33 * 0,66 = 0.01349315. Probabilidade NO: P(no) * P(sunny |no) * P(mild|No) * P High|no) * P(FALSE|no), 0,35 * 0,6 * 0,4 * 0,8 * 0,4 =0.02688. Portanto, a classe No vence com 0,02688

### p62

Marque as opções abaixo que são métricas utilizadas para avaliar a qualidade dos agrupamentos criados por um agrupador (marque todas que se aplicam):

+ IaCD (Intra-Cluster DIstance)
+ IeCD (Inter-CLuster Distance)

Explicação
IeCD e IaCD são métricas para avaliar os clusters criados

### p70

Princípio de que a probabilidade de mudança de estado depende apenas do estado atual, e não dos estados anteriores:

Cadeias de Markov

Explicação
Cadeias de Markov são os princípios básicos de sistemas de aprendizado como Reinforcement Learning

### p71 

Você criou um modelo de machine learning cujo índice de acertos no primeiro treino/teste foi de 90%.

Como seu modelo foi criado a partir de amostras e está sujeito a variações dos dados e do modelo de negócio, você decide calcular o intervalo de confiança para a média dos acertos do modelo e para isso você treina e testa um total de 100 modelos. Destes 100 modelos o desvio padrão foi igual a 12,61 e a média da precisão foi de 88,2.

Qual é o intervalo de confiança para o acerto médio do modelo para um nível de confiança de 95%?

> Entre 85,7 e 90,67

Explicação
z = 1,96 (95%) dp = desvio padrão x +- z * (dp / raiz(n))x +- 1,96 * (12,61/raiz(100))x +- 2,47. Como a média é 88,2 , a variação esperada é de 85,7 até 90,67 em 95% das vezes.

### p72

Na imagem abaixo, temos 2 relações. Na primeira relação, temos as classes no 5º atributo, sendo elas Iris-versicolor, Iris-virginica e iris-setosa.  Na segunda relação, temos as classes nos 6 primeiros atributos, sendo estes atributos amazed-suprised,  happy-pleased, relaxing-clam, quiet-still, sad-lonely  e angry-aggresive, sendo que cada instância pode receber apenas os valores zero ou um.

Do ponto de vista de machine learning, que tipo de classes tem estas duas relações, respectivamente?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_19-40-40-86a444619de82a6cefef288275e6c9b5.png

> Multilabel e MUltiClass

Explicação
Quando a tarefa de classificação é para classificar a instância em apenas uma única classe, dizemos que é multilabel. Quando ela deve ser feita em mais de uma classe, dizemos que ela é multiclass

### p75

Qual opção abaixo é uma técnica para reduzir o efeito da chamada maldição da dimensionalidade em um modelo estatístico?

> Seleçâo de subconjntos de atributos

Explicação
A seleção de atributos é um processo para selecionar um subconjunto de atributos relevantes para determinado problema

### p80

Em um agrupador k-means, como é feita a atualização dos centroides?

Pela média das instnacias que fazem aprte do grupo

Explicação
Embora a forma de cálculo possa ser pela distância euclidiana, a atualização do centroide se da pela média das instâncias que fazer parte do grupo. Dessa forma, o centroide pode mudar de posição e as instâncias podem sofrer alterações em seus grupos

### p87

A fórmula da imagem abaixo é do cálculo de entropia. Você tem um conjunto de dados com 20 instâncias. Temos 2 classes, saudável com 13 instâncias e doente com 7 instâncias. Calcule a entropia da classe destes dados.

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-28_17-24-37-d36e0dd424986413a1a4aef80ba7ff65.PNG

0,9340681

Explicação
Aplicando a fórmula da entropia: (-13/20*log2(13/20))+(-7/20*log2(7/20)) = 0,9340681

### p88

Marque a alternativa correta sobre Active Learning:

Dados não classificados são exibidos ao usuário que de forma interativa faz a classificação

### p89

Sobre a técnica não supervisionada denominada Gaussian Mixture Models (GMM), marque a opção correta:

Gera conjuntos de agrupamentos dos dados originais a partir de uma distribuição normal

Explicação
Embora os grupos sejam gerados a partir de uma distribuição gaussiana, eles são produzidos a partir dos dados originais

### p99

Qual opção abaixo melhor define a técnica denominada imputação?

Processo de substituir valores faltantes em um conjunto de dados

Explicação
Imputação é o processo de substituir valores faltantes em conjuntos de dados, muitas vezes porque determinados classificadores não suportam valores faltantes. Existem diferentes técnicas, que são avaliadas caso a caso

