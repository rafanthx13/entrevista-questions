# Level Normal

## Assuntos

## Medidas de centralidade e variabilidade

### p02

Marque a opção verdadeira:

O desvio padrãoe'a raiz quadrada da variancia

Explicação
O cálculo do desvio padrão é feito através da raiz quadrada da variância

### p24

Qual quantil é equivalente a mediana?

Segundo

Explicação
O segundo quantil, ou Q2, equivale a mediana

### p51

Quanto um variável tem um número par de valores, como é calculada a mediana?

Calculando-se a média dos dois valores centrais

Explicação
Para este caso basta ordenar os dados de forma crescente, e calcular a média dos dois valores que estarão ao centro

### p56

A variância possui cálculos diferentes para amostra e população. Qual é diferença neste cálculo?

Na amostra, a divisão se dá por n-1

Explicação
Tanto para a variância quanto para o desvio padrão da amostra, a divisão se da por n-1

---
---
---

## Regressão Linear

### p53

Se um modelo de regressão linear tem correlação (R) = 1, isso significa que:

A soma de seus resíduos será igual a zero

Explicação
Uma correlação perfeita indica que todos os valores de treino coincidiriam com a linha de melhor ajuste, portanto, os resíduos, que é diferença entre os dados e a linha, seria igual a zero, inexistente

### p64

Marque a opção falsa com relação a regressão logística

A variável de reposta é contínua

Explicação
Embora a regressão logística produza uma previsão em termos de probabilidade, a variável de resposta é categórica

### p66

Por que se em um modelo não temos resíduos próximos a uma distribuição normal, temos indícios de que não temos um bom modelo?

Porque provavelmente informações que deveriam estar no modelo, ficaram nos resíduos

Explicação
Um modelo cujos resíduos não estão próximos a uma distribuição normal, podem indicar que ainda existam informações que deveriam estar no modelo, não nos resíduos

### p67

Marque a afirmação falsa sobre regressão logística:

As variáveis  independente devem ser obrigatoriamente categóricas

Explicação
Não há obrigatoriedade das variáveis independentes serem categóricas em regressão logística

### p68

Qual opção abaixo é um bom valor para um coeficiente de determinação (R ao quadrado)?

0.7

Explicação
O coeficiente de determinação é um valor positivo entre zero e um. Das opções, a única possível é 0,7, que é um bom valor para o parâmetro

### p79

Na regressão linear múltipla, temos duas ou mais variáveis dependentes para explicar uma variável explanatória. A afirmação acima é:

Falsa

Explicação
O sentença esta invertida, variáveis independentes que explicam uma variável dependente

### p90

Quando mais rápido gira um moinho de vento, mais vento se observa. Portanto, o vento é causado pela rotação dos moinhos de vento. Como é conhecida esta falácia lógica em estatística?

Correlaçâo nâo implica causa

Explicação
A afirmação é um caso clássico de que, mesmo que exista uma relação matemática entre dois fenômenos, não necessariamente um causou o outro

### p91

Você constrói um modelo de regressão linear e observa que a correlação é de apenas 0,3. Analisando os dados, você nota que existe um valor bem acima dos demais valores. Você elimina este valor e o índice de correlação agora é de 0,8. Você faz uma investigação e obtêm fortes indícios de que este valor fora do padrão foi gerado devido a uma falha no processo de coleta. O que você faria?

Mantenho o modelo sem o dado fora do padrâo


Explicação
Não existe uma regra universal para tratamento de dados fora do padrão, porém, se temos indícios fortes que ele foi introduzido devido a um erro no processo, o mais razoável a fazer é eliminar este dado e utilizar o modelo sem ele

### p94

Imagine um problema de regressão linear. O valor da variável no eixo Y somado ao resíduo gerado a partir da criação do modelo, é igual a:

Valor ajustado, ou valor da linha de regressão

Explicação
A definição é de valor ajustado, ou seja, o ponto equivalente dos dados de treino na linha de regressão

---
---
---

## PNL/NLP (Processamento de LInguagem Natural/Natural Porcessing Learning)

### p31

Processo em Processamento de Linguagem Natural, que consiste em registrar elementos de composição do texto, como flexões e dependências.

Annotations

Explicação
O processo é chamado de Annotations, ou anotações, onde anotações sobre a composição do texto são adicionadas

### p61

Em processamento de linguagem natural, é um formato  CoNLL de padrão para annotations:

CoNLL-U

Explicação
CoNLL-U é um formato, ou modelo padrão, para processamento de linguagem natural

### p93

São aplicações tipicas de Processamento de Linguagem Natural (marque todas que se aplicam):

Traduçâo, reochecimento de Fala, Correçâo Ortográfica, Análise de SENTIMENTOS

Explicação
Todas as opções são tarefas tipicas de processamento de linguagem natural

---
---
---

## Neural Net e Deep Learning

### p06

Em Reinforcement Learning, qual opção melhor define as "policies"?

Os próximos estado otimizados, a partir do estado atual

Explicação
Reinforcement Learning busca a mudança otimizada de estados. As policies possuem os melhores politicas de mudanças de estado, de acordo com o modelo criado no aprendizado


### p08

Em machine learning, qual o principal objetivo de se executar normalização de dados?

Eiliinar dados redundantes

Explicação
A normalização busca reduzir dados redundantes

### p15

Em redes neurais artificias, qual opção é a melhor definição para Epoch?

Uma passagem pro todos os dados de treino

Explicação
Epoch é a passagem de todos os registros pela rede neural para ajuste de pesos (treino)

### p25

Em uma rede neural artificial, o que define se o neurônio artificial vai emitir um "sinal" e ativar ou não o próximo neurônio da rede?

A função de ativação

Explicação
A função de ativação serve exclusivamente para receber um valor da rede, processar este valor e de acordo com a saída, ativar ou não o restante da rede

### p35

Das alternativas abaixo, qual é a melhor definição para Backpropagation?

Cálculo de erro que é utilizado para ajustar os pesos nas entradas para tnetar melhorar a perfomance da rede

???

### p57

Marque a opção que tem o número correto de camadas ocultas da rede neural da imagem abaixo

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_23-53-49-2b421bb640223c94045523a66ffeb858.png

3

Explicação
Temos três camadas ocultas, que pode ser identificados pelas filas de neurônios artificiais vermelhos. Camadas ocultas soa aquela que nâo são nem a última e nem a primeira de entrada

### p58

Qual a melhor definição para a topologia de uma rede neural artificial?

Sua estrutura de nós e camadas

Explicação
A topologia diz respeito a estrutura de nós e camadas. Feed Forward ou recorrente se refere a arquitetura da rede

### p63

Em redes neurais artificias, qual opção é a melhor definição para Iterações?

Cada passagem de todos os registros do batch

Explicação
Uma vez definido o tamanho de batch, a passagem do batch pela rede neural para ajuste de pesos (treino) é chamado de iteração

### p81

Em redes neurais artificias, qual opção é a melhor definição para batch size?

Total de registros que serão passados em uma iteraçâo

Explicação
Iteração é a passagem de registros definidos no bacth, então batch size é o número de registros que vão compor esta iteração

### p84

Em redes neurais artificiais, qual opção melhor define o aprendizado online?

Pesos do modelo são atualizados ao final da iteração

Explicação
No aprendizado online os pesos são atualizados a cada instância, já no modo batch os pesos só são atualizados após todas as instâncias

### p86

Qual opção melhor descreve o funcionamento de Convolutional Neural Networks?

No processo de aprendizado, características dos objetos formam filtros, que na classificação são utilizados para buscar estas mesmas características em novas imagens. Quanto mais características de um objeto forem identificadas, maiores as chances de classificação com aquela classe.

Explicação
Redes convolucionais funcionam a partir de características dos objetos, e não através do treino da rede neural de acordo com cada bit da imagem

### p95

Qual opção abaixo é a melhor definição para Epochs?

Uma passagem por todos os dados de treino

Explicação
Uma epoch e uma unica passagem por todos os dados de treino do modelo

---
---
---

## Machine Learning

### p05

Um problema de machine learning foi implementado usando um algoritmo Naive Bayes. As classes existentes são Modelo1, Modelo1 e Modelo3. Uma vez construído o modelo, uma instância de teste foi submetida a ele, chegando-se aos seguintes resultados para a probabilidade posterior: Modelo1 = 0,004, Modelo2 = 0,00567 e Modelo3 = 0,0005. Baseado neste resultado, com qual classe e porque a instância deve ser classificada?

Como Modelo2, por ter a prob posterior mais alta


Explicação
Naive Bayes calcula a probabilidade posterior e classifica de acordo com a classe que tiver a probabilidade mais alta

### p07

São técnicas que podem ser utilizadas para criar modelos de machine learning com melhor precisão, com exceção de:

Aumentar capacidade de processamento (CPU)

Explicação
Aumentar a CPU pode trazer melhor desempenho, mas não vai aumentar a taxa de precisão ou de acertos do modelo. Você pode testar classificadores com configurações mais "pesadas" e assim ter um melhor modelo, mas essa seria uma vantagem indireta

### p09

Técnica de classificação que ajusta os pesos das instâncias baseado no acerto ou erro da classe durante o treino:

Boosting

Explicação
O enunciado descreve o funcionamento básico da técnica conhecida como boosting, onde, basicamente, são atribuídos pesos as classes de acordo com erros ou acertos

### p13

Comparando classificadores baseados em modelos com classificadores baseados em instância, qual é a melhor opção com relação a vantagens e desvantagens de ambos os modelos?

Modelos tem mais custo pré-processamento. Baseados em instância tem maior custo no processo de classificação

Explicação
Existem modelos que são de difícil interpretação para humanos, como por exemplo, RNA, mas isso não é uma regra. Por isso, a melhor resposta é custo pré-processamento para modelos e custo na classificação para os baseados em instâncias

### p14

Técnica de agrupamentos (Clusters) em que o número de grupos deve ser definido como parâmetro pelo usuário:

K-means

Explicação
K-means e K-medoids precisam que o número de clusters seja definido, ou seja, o parâmetro não é definido automaticamente pelo algoritmo

### p17

Qual opção abaixo é a melhor definição para o fenômeno conhecido como curse of dimensionality?

Muitas dimensões (atributos) podem deteriorrar a perfomance de um mdoelo

Explicação
Um modelo com muitas características tende a ficar com a performance degradada

### p18

São técnicas válidas de classificação multilabel (marque todas que se aplicam):

Adaptaçâo de algoritmos; Tranformação de problema

Explicação
Clare e Adaboost são algoritmos adaptados, portanto, a resposta correta são Transformação de Problema e Adaptação de Algoritmos

### p19

Qual afirmação abaixo é falsa com relação ao algoritmo de agrupamentos DBSCAN?

Agrupa todos os elementos e nâo gera ruídos (Não agrupados)

Explicação
DBSCAN pode eventualmente não agrupar elementos, o que é considerado ruído

### p20

As três árvores de decisão da imagem abaixo foram induzidas a partir dos mesmos dados de treino. Marque a opção falsa com relação a elas.

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_23-18-57-8ea1f8d8540522e035f0371916d30af4.PNG

O desempenho da árvore mais a direita, deve ser melhor do que a árvore ao centro

Explicação
Podemos afirmar que as árvores sofreram um processo de poda ou que foi definido um número mínimo de instâncias para partição. Porém, sem testar o modelo, não podemos afirmar que o desempenho de qualquer uma delas é melhor do que outro.

### p21

Observando a tabela de probabilidades da imagem abaixo, onde a classe é o atributo Play, qual é a opção mais correta?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_23-15-06-fa98de163ab0441d4496d352fe3d2a72.png

Trata-se de um modelode ML cuja dependencia exte entre os atributos e a classe, e nao dos atributos entre si

Explicação
Claramente existe uma estrutura de probabilidade condicional, pois temos mais de um atributo condicionando a classe

### p23

Qual é o princípio fundamental do algoritmo de associação apriori:

Se um conjunto é frequente, um subconjunto também é

Explicação
Embora apriori possa criar regras com um suporte e confiança mínimos, não é correto dizer que só são criadas regras para maiores suportes e confianças. O principio básico do funcionamento do algoritmo é a opção 4

### p27

Você deve desenvolver um modelo para um sistema de vendas de varejo, com o objetivo de minimizar fraudes. Você treinou dois modelos com classificadores diferentes. Embora a taxa de acertos tenha sido semelhante nos dois modelos, o primeiro modelo minimizou falsos positivos, já o segundo, minimizou falsos negativos. Qual destes modelos, provavelmente, o cliente deve querer colocar em produção?

O primeiro modelo

Explicação
Falsos positivos neste caso significam perdas de dinheiro, pois a transação será validada mas não era uma transação legítima. Já falsos negativos significa perda de oportunidades, são vendas legítimas que o sistema entende que são fraudes. Em geral, para este caso, é melhor evitar perda de dinheiro do que perder oportunidades.

### p29

Em machine learning, qual opção apresenta a melhor definição para discretização de atributos?

Tranformar valores contínuos em categóricos

Explicação
Discretização é o processo de criar categorias, por exemplo, idades podem ser transformados em criança, adolescente, adulto e idoso

### p32

Qual a melhor definição para dados desbalanceados?

Poucos dados de determinandas classes

Explicação
Dados desbalanceados são quando existem poucos dados de uma determinada classe (ou mais de uma), o que dificulta a criação de um modelo genérico

### p36

Qual é o princípio fundamental do algoritmo de associação FP-Grow:

Induz árvores, e busca sobreposição destas árvores, onde os intes sâo frequentes

Explicação
O principio de itens frequentes e subconjunto frequentes é do algoritmo apriori. FP-Grow induz árvores e busca onde existem sobreposições

### p39

Como garantir que em um processo de classificação com KNN não ocorra um empate?

Definindo k como um número ímpar

Explicação
K é o parâmetro que representa o número de vizinhos que serão selecionados. Se o número for impar, não tem como haver empate no processo

### p40

Conjunto de classificadores independentes podem ter uma melhor performance do que um único classificador, são os classificadores que usam métodos de grupos. Dos métodos utilizados por estes tipos de classificadores, marque a opção falsa:

Classificam utilizando diferentes técnicas de classificaçâo, por exemplo, Decision Tree seguidos por redes Bayseanas

Explicação
Métodos de grupos não usam diferentes algorítimos, mas sim um mesmo classificador, alterando parâmetros, dados ou atributos

### p41

Marque a opção falsa com relação ao classificador KNN:

KNN cria sesmpre modelos com boa perfomance computacional

Explicação
KNN é um classificador baseado em instância, não em modelos. Normalmente a performance é pior do que os classificadores baseados em modelos, por isso esta é a opção falsa

### p42

Técnica de machine learning que utiliza um hiperplano para separar diferentes classes no processo de criação do modelo.

SVM

Explicação
Apenas Máquina de Vetor de Suporte (SVM) utiliza hiperplanos para separação de classes

### p43

Qual das alternativas abaixo melhor define a profundidade de uma árvore de decisão?

Número máximo de nodos da raiz até a folha

Explicação
A profundidade é a medida vertical da árvore de decisão, da sua raíz até suas folhas

### p55

Observe a árvore de decisão da imagem abaixo. Das alternativas abaixo, qual tem a melhor justificativa para o atributo outlook ter sido escolhido, pelo algoritmo que construiu esta árvore, como nó raiz?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_23-16-21-2e9399e64844303a5111ec2b19fce485.PNG

O atributo teev o maior valor a ser calculado o ganho de informação


Explicação
Embora a entropia seja importante e utilizada no cálculo, o que define a maior pureza para a escolha do atributo raiz é o cálculo do ganho de informação

### p59

Considerando transações A e B e  B e A Qual alternativa podemos afirmar que é verdadeira?

O suporte das duas transações deve ser o mesmo

Explicação
Pela definição de suporte, a inversão das transações vai trazer o mesmo indíce

### p87

Com relação a K-means, marque a opção falsa:

É um classificador baseado em instância

Explicação
K-means não é um classificador, mas sim um algorítimos de agrupamentos (Clusters), por isso esta opção é falsa

### p92

Induz múltiplas estruturas de árvores de decisão, utilizando particionamento por bootstrap e selecionando subconjuntos de atributos. A qual opção abaixo se refere esta técnica de classificação?

Random Forest

Explicação
O processo descrito é o funcionamento de uma Random Forest

### p98

Dada a matriz de confusão da imagem abaixo, qual é taxa de acertos do modelo?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_23-38-43-05df6c164f4fab4ef2959234df78f69a.PNG

0.84

Explicação
Acertos são calculados pela fórmula (VP+VN)/Total, (2980+1253)/(2980+442+325+1253) = 0,8466

---
---
---

## Anova

### p88

Quando devemos utilizar a analise de variância (Anova) em vez de um Teste T para analisar a variação de médias?

Quando temos mais de dois grupos

Explicação
Para dois grupos usamos o Teste T. Para mais de dois grupos, Anova.

---

## Séries Temporias

### p01

Um modelo Arima é composta pelos parâmetros (p, d, q). Qual opção equivale ao parâmetro d?

Grau de diferenciação

Explicação
O parâmetro d é o grau de diferenciação aplicado a série

### p04

Qual a principal diferença entre a Tendência Linear de Hold e a Tendência Amortecida de Hold?

Na tendência amortecida de hold, a tendẽncia é amortecida de acordo com o parâmetro FI, conforma avana para o futuro. Já a tendência linear de Hold crescen ao infinito

Explicação
A tendência amortecida não cresce de forma linear ao infinito, ela é amortecida de acordo com o parâmetro FI

### p26

Você executou análises gráficas de resíduos de um modelo de série temporal. Esta análise pode ser vista na imagem abaixo. O que podemos concluir com esta análise?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_20-17-01-915939f1ecdd8bf8e4b9bf7793932721.png

Os resíduos nâo são autocorrelacionados

Explicação
O primeiro gráfico mostra ausência de padrão. O gráfico de autocorrelação mostra ausência de autocorrelação. O histograma mostra uma curva próxima de sino, esperado para dados normalmente distribuídos. Então, temos fortes indícios de que os resíduos não estão autocorrelacionados e temos um bom modelo

### p28


Técnica de previsão de séries temporais onde o último valor é projetado para o futuro.

Naive

Explicação
Naive é uma técnica extremamente simples que projeta o último valor para o futuro


### p34

Dois modelos preditivos de séries temporais geraram os índices abaixo:

Primeiro modelo:

AIC=1017.85   AICc=1018.17   BIC=1029.35

Segundo Modelo:

AIC=900.55   AICc=1000.15   BIC=1029.35

Segundo estes indicies, qual seria o melhor modelo?

Segundo modelo

Explicação
O segundo modelo tem os menores indíces de AIC e AICs, sendo que BIC ficou empatado.

### p48

Aplicada a séries temporais, Box-Cox é um processo de:

Tranformação de séries temporais

Explicação
Box-Cox é um processo para transformar séries temporais


### p65

No método de suavização exponencial, qual são os parâmetros possíveis para o elemento sazonal? (marque todos que se aplicam)

Aditivo; Multiplicativo

Explicação
O método de suavização exponencial pode ser aditivo ou multiplicativo


### p69

Observe o gráfico na imagem abaixo. Qual técnica de previsão de séries temporais foi utilizada?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_22-31-01-86d5d32f8b29d431700d9f517b89e55b.png

Drift

Explicação
Naive não pode ser pois ela simplesmente projeta o valor para o futuro. Mean projeta a média para o futuro, e portanto não haveria tendência na previsão. Das técnicas listadas drift é a única que acompanha a tendência da série, portanto esta é a resposta correta.

### p70

Qual o principal efeito da aplicação de médias móveis em uma série temporal?

Suavizaçâo da série

Explicação
Embora as médias móveis possam reduzir efeitos sazonais, seu principal efeito aplicado a uma série temporal é suavizar a série e eliminar outliers

### p71

Qual a principal razão para executarmos um processo de diferenciação em uma Série Temporal?

Tranformar uma série temporal não estacionária em estacionária, a fim de melhorar o processo de forecast

Explicação
Alguns modelos preditivos de séries temporais exigem séries estacionárias. A diferenciação é capaz de transformar uma série não estacionária em estacionária

### p75

São objetivos possíveis na aplicação de médias móveis em series temporais (marque todas que se aplicam):

Suavizaçâo da série; REmoção de outiliers; Identificação de tendências; Preparação para processos de forecast

Explicação
Todas as alternativas são plausíveis quando aplicamos médias móveis em uma série temporal

---
---
---

## Estatística

### p11

Em um boxplot, a linha central da caixa (linha em vermelho no gráfico abaixo) é qual parâmetro estatístico?

https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-27_20-06-59-31c4762393e9f9b9b4c9229f556070c4.png

Mediana

Explicação
Tradicionalmente a linha é a mediana da variável

### p12

Sobre a distribuição normal padrão (Z), marque a alternativa correta:

Tem média igual a zero e desvio padrão igual a 1

CONTINUA  na p30 ..........

### p30

Qual a principal função da transformação de Box Cox?

Transformar dados de uma variável dependente para que sejam normalmete distríbuidos

Explicação
A transformação de Box Cox tem como principal função transformar dados não normalmente distribuídos em dados que estejam aproximadamente normalmente distribuídos

### p49

Entre as razões para a execução de processos de Análise de Dados Exploratória, qual opção seria a MAIS importante?

Compreender os dados e os fenomoenos que o produziram

Explicação
O principal objetivo da EDA é compreender um fenômeno através de dados, para isso podemos utilizar técnicas de visualização bem como resumos estatísticos

### p73

Mean Erro (ME) é uma métrica de avaliação de performance utilizada em regressão e séries temporais. Qual é um problema específico desta métrica?

A métrica pode ser anulada ou distorcida com a presença de valores positivos e negativos

Explicação
O fato da métrica ser dependente de escala é apenas uma característica, sua principal fraqueza está no fato de seu resultado poder se anular na presença de valores negativos e positivos. Deve ser usada apenas para valores sempre positivos ou sempre negativos, ou substituída por MAE

### p78

Os dois principais tipos de variáveis aleatórias são as discretas e as contínuas. Sobre a diferença das duas, marque a opção correta:

Variaveis continuas mede, variaveis discretas contam

Explicação
A melhor definição é a número 1. Embora em uma distribuição normal se espere encontrar valores contínuos, dados discretos também pode estar normalmente distribuídos

### p82

Marque a opção que apresenta a melhor definição para percentil:

Percentual de dados que estao abaixo de determinado valor

Explicação
Das definições apresentadas, a melhor é de dados percentualmente encontrados abaixo de determinado valor

### p83

A estatística descritiva representa dados através do resumo dos cinco números. Das opções abaixo, qual parâmetro não faz parte deste resumo?

Média

Explicação
Os parâmetros são: mínimo, quartil inferior, mediana, quartil superior e máximo. Portanto a média não faz parte

### p89

Sobre a estatística descritiva, marque a melhor resposta:

Tem por objetivo demonstrar dadods através de resumos e gráficos

Explicação
Todas as opções são válidas para processos estatísticos, mas apenas a primeira opção descreve e estatística descritiva


