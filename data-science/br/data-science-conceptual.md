---
layout: default
title: Data Science PT-BR
description: Questões de DataScience e Machine Learning
---

## Índex

## Links
+ [https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo](https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo)

## Glossário

ALguns termos preferi manter em ingles

+ dataset: o conjunto de dados
+ rows: linhas, no caso os registro da dataset
+ 

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

link:  https://www.datamation.com/big-data/data-mining-techniques.html

Data analytics and the growth in both structured and unstructured data has also prompted data mining techniques to change, since companies are now dealing with larger data sets with more varied content. Additionally, artificial intelligence and machine learning are automating the process of data mining.

Regardless of the technique, data mining typically evolves over three steps:

+ Exploration: First you must prepare the data, paring down what you need and don’t need, eliminating duplicates or useless data, and narrowing your data collection to just what you can use.
+ Modeling: Build your statistical models with the goal of evaluating which will give the best and most accurate predictions. This can be time-consuming as you apply different models to the same data set over and over again (which can be processor-intensive) and then compare the results.
+ Deployment: In this final stage you test your model, against both old data and new data, to generate predictions or estimates of the expected outcome.
Leading Data Mining Techniques
Data mining is an highly effective process – with the right technique. The challenge is choosing the best technique for your situation, because there are many to choose from and some are better suited to different kinds of data than others. So what are the major techniques?

Classification Analysis

This form of analysis is used to classify different data in different classes. Classification is similar to clustering in that it also segments data records into different segments called classes. In classification, the structure or identity of the data is known. A popular example is e-mail to label email as legitimate or as spam, based on known patterns.

Clustering

The opposite of classification, clustering is a form of analysis with the structure of the data is discovered as it is processed by being compared to similar data. It deals more with the unknown, unlike classification.

Anomaly or Outlier Detection

This is the process of examining data for errors that may require further evaluation and human intervention to either use the data or discard it.

Regression Analysis

A statistical process for estimating the relationships between variables which helps you understand the characteristic value of the dependent variable changes. Generally used for predictions, it helps to determine if any one of the independent variables is varied, so if you change one variable, a separate variable is affected.

Prediction/Induction Rule

This technique is what data mining is all about. It uses past data to predict future actions or behaviors. The simplest example is examining a person’s credit history to make a loan decision. Induction is similar in that it asks if a given action occurs, then another and another again, then we can expect this result.

Summarization

Exactly as it sounds, summarization present a mode compact representation of the data set, thoroughly processed and modeled to give a clear overview of the results.

Sequential Patterns

One of the many forms of data mining, sequential patterns are specifically designed to discover a sequential series of events. It is one of the more common forms of mining as data by default is recorded sequentially, such as sales patterns over the course of a day.

Decision Tree Learning

Decision tree learning is part of a predictive model where decisions are made based on steps or observations. It predicts the value of a variable based on several inputs. It’s basically an overcharged "If-Then" statement, making decisions on the answers it gets to the question it asks.

Tracking Patterns

This is one of the most basic techniques in data mining. You simply learn to recognize patterns in your data sets, such as regular increases and decreases in foot traffic during the day or week or when certain products tend to sell more often, such as beer on a football weekend.

Statistical Techniques

While most data mining techniques focus on prediction based on past data, statistics focuses on probabilistic models, specifically inference. In short, it’s much more of an educated guess. Statistics is only about quantifying data, whereas data mining builds models to detect patterns in data.

Visualization

Data visualization is the process of conveying information that has been processed in a simple to understand visual form, such as charts, graphs, digital images, and animation. There are a number of visualization tools, starting with Microsoft Excel but also RapidMiner, WEKA, the R programming language, and Orange.

Neural Networks

Neural network data mining is the process of gathering and extracting data by recognizing existing patterns in a database using an artificial neural network. An artificial neural network is structured like the neural network in humans, where neurons are the conduits for the five senses. An artificial neural network acts as a conduit for input but is a complex mathematical equation that processes data rather than feels sensory input.

Data Warehousing

You can’t have data mining without data warehousing. Data warehouses are the databases where structured data resides and is processed and prepared for mining. It does the task of sorting data, classifying it, discarding unusable data and setting up metadata.

Association Rule Learning

This is a method to identify interesting relations and interdependencies between different variables in large databases. This technique can help you find hidden patterns in the data that that might not otherwise be clear or obvious. It’s often used in machine learning.

Long-Term Memory Processing

Data processing tends to be immediate and the results are often used, stored, or discarded, with new results generated at a later date. In some cases, though, things like decision trees are not built with a single pass of the data but over time, as new data comes in, and the tree is populated and expanded. So long-term processing is done as data is added to existing models and the model expands.

Data Mining Best Practices

Regardless of which specific technique you use, here are key data mining best practices to help you maximize the value of your process. They can be applied to any of the 15 aforementioned techniques.

Preserve the data. This should be obvious. Data must be maintained militantly, and it must not be archived, deleted, or overwritten once processed. You went through a lot of trouble to get that data prepared for generating insight, now vigilance must be applied to maintenance.
Have a clear idea of what you want out of the data. This predicates your sampling and modeling efforts, never mind your searches. The first question is what do you want out of this strategy, such as knowing customer behaviors.
Have a clear modeling technique. Be prepared to go through many modeling prototypes as you narrow down your data ranges and the questions you are asking. If you aren’t getting the answers you want, ask them a different way.
Clearly identify the business problems. Be specific, don’t just say sell more stuff. Identify fine grain issues, determine where they occur in the sale, pre- or post-, and what the problem actually is.
Look at post-sale as well. Many mining efforts focus on getting the sale but what happens after the sale -- returns, cancellations, refunds, exchanges, rebates, write-offs – are equally important because they are a portent to future sales. They help identifying customers who will be more or less likely to make future purchases.
Deploy on the front lines. It’s too easy leave the data mining inside the corporate firewall, since that’s where the warehouse is located and all data comes in. But preparatory work on the data before it is sent in can be done in remote sites, as can application of sales, marketing, and customer relations models.

___
----
---

## DIF DATA SNCEINCE FOMR THER REST


Data is almost everywhere. The amount of digital data that currently exists is now growing at a rapid pace. The number is doubling every two years and it is completely transforming our basic mode of existence. According to a paper from IBM, about 2.5 billion gigabytes of data had been generated on a daily basis in the year 2012. Another article from Forbes informs us that data is growing at a pace which is faster than ever. The same article suggests that by the year 2020, about 1.7 billion of new information will be developed per second for all the human inhabitants on this planet. As data is growing at a faster pace, new terms associated with processing and handling data are coming up. These include data science, data mining and machine learning. In the following section- we will give you a detailed insight on these terms. 

What is data science?

Data Science deals with both structured and unstructured data. It is a field that includes everything that is associated with the cleansing, preparation and final analysis of data. Data science combines the programming, logical reasoning, mathematics and statistics. It captures data in the most ingenious ways and encourages the ability of looking at things with a different perspective. Likewise, it also cleanses, prepares and aligns the data. To put it more simply, data science is an umbrella of several techniques that are used for extracting the information and the insights of data. Data scientists are responsible for creating the data products and several other data based applications that deal with data in such a way that conventional systems are unable to do.

What is data mining? 

Data mining is simply the process of garnering information from huge databases that was previously incomprehensible and unknown and then using that information to make relevant business decisions. To put it more simply, data mining is a set of various methods that are used in the process of knowledge discovery for distinguishing the relationships and patterns that were previously unknown. We can therefore term data mining as a confluence of various other fields like artificial intelligence, data room virtual base management, pattern recognition, visualization of data, machine learning, statistical studies and so on. The primary goal of the process of data mining is to extract information from various sets of data in an attempt to transform it in proper and understandable structures for eventual use. Data mining is thus a process which is used by data scientists and machine learning enthusiasts to convert large sets of data into something more usable.

What is machine learning? 

Machine learning is kind of artificial intelligence that is responsible for providing computers the ability to learn about newer data sets without being programmed via an explicit source. It focuses primarily on the development of several computer programs that can transform if and when exposed to newer sets of data. Machine learning and data mining follow the relatively same process. But of them might not be the same. Machine learning follows the method of data analysis which is responsible for automating the model building in an analytical way. It uses algorithms that iteratively gain knowledge from data and in this process; it lets computers find the apparently hidden insights without any help from an external program. In order to gain the best results from data mining, complex algorithms are paired with the right processes and tools. 

What is the difference between these three terms?

As we mentioned earlier, data scientists are responsible for coming up with data centric products and applications that handle data in a way which conventional systems cannot. The process of data science is much more focused on the technical abilities of handling any type of data. Unlike data mining and data machine learning it is responsible for assessing the impact of data in a specific product or organization. 

While data science focuses on the science of data, data mining is concerned with the process. It deals with the process of discovering newer patterns in big data sets. It might be apparently similar to machine learning, because it categorizes algorithms. However, unlike machine learning, algorithms are only a part of data mining. In machine learning algorithms are used for gaining knowledge from data sets. However, in data mining algorithms are only combined that too as the part of a process. Unlike machine learning it does not completely focus on algorithms. 

https://www.datasciencecentral.com/profiles/blogs/difference-of-data-science-machine-learning-and-data-mining?fbclid=IwAR2HY7zR7HIGm19vXzWhH84UQdk3neilbi0FKa14Vyd8y0C0ZixH8tmZmcg



-----
----
-----

## BUSCA DE EMPREGO

http://datascienceacademy.com.br/blog/12-licoes-aprendidas-em-entrevistas-para-cientista-de-dados/

----
----
-----

## PLENO, JUNIRO, SENIOR

http://www.cienciaedados.com/cientista-de-dados-junior-pleno-e-senior/

Cientista de Dados Junior – Entre no Nível 1.0
Cientista de Dados Júnior em geral é graduado, completou alguns cursos online de especialização e está iniciando sua caminhada em Data Science. Áreas de estudo populares incluem Ciência da Computação, Matemática ou Engenharia (embora outras áreas de formação sejam inteiramente possíveis). Um Cientista de Dados Júnior tem 0 a 2 anos de experiência profissional e está familiarizado com a criação de protótipos com conjuntos de dados estruturados em Python ou R. Ele(a) participou de competições no Kaggle e possui um perfil no GitHub.

O Cientista de Dados Júnior pode fornecer um enorme valor para as empresas. Eles geralmente são autodidatas, já que poucas universidades oferecem diplomas em Ciência de Dados e, portanto, mostram tremendo comprometimento e curiosidade, usando cursos online como aprendizado complementar e focado. Eles estão entusiasmados com o campo que escolheram e estão ansiosos para aprender mais. O Cientista de Dados Júnior é bom em prototipagem de soluções, mas ainda não possui proficiência na mentalidade de negócios.

O Cientista de Dados Júnior deve ter uma forte paixão por Machine Learning. Você pode demonstrar sua paixão contribuindo para projetos de código aberto, participando de desafios do Kaggle e criando seu portfólio.

O que eles fazem?
Se uma empresa está contratando um Cientista de Dados Júnior, geralmente já existe uma equipe de Data Science. A empresa está procurando ajuda para facilitar a vida de colegas mais experientes. Isso envolve testar rapidamente novas ideias, depurar e refatorar modelos existentes. O profissional discutirá ideias com a equipe e lançará novas ideias sobre como fazer as coisas melhor. Assume a responsabilidade pelo seu código, buscando continuamente melhorar a qualidade e o impacto do código. O profissional sabe trabalhar em equipe e está buscando constantemente apoiar seus colegas na missão de criar ótimos produtos de dados.

O que eles não fazem?
O Cientista de Dados Júnior não tem experiência em engenharia de soluções complexas de produtos. Portanto, ele(a) trabalha em equipe para colocar em produção modelos de Ciência de Dados. Como o Cientista de Dados Júnior acabou de ingressar na empresa, ele(a) não está imerso nos negócios da empresa. Portanto, não se espera que ele(a) apresente novos produtos para impactar a Equação Fundamental dos Negócios. No entanto, o que sempre é esperado é o desejo de aprender e melhorar suas habilidades.

Todo início de carreira é complicado pois somente a experiência do dia a dia permite consolidar conhecimentos teóricos, mas com muita vontade de aprender, dedicação e pró-atividade para buscar o conhecimento e praticar, os primeiros passos serão mais fáceis.

Cientista de Dados Pleno – Atingindo o Nível 2.0
O Cientista de Dados Pleno já trabalhou como Cientista de Dados Júnior e tem de 2 a 5 anos de experiência relevante, escreve códigos reutilizáveis e constrói pipelines de dados resilientes em ambientes em nuvem.

O Cientista de Dados Pleno deve ser capaz de enquadrar os problemas da Ciência de Dados. Bons candidatos têm ótimas ideias de experiências anteriores em Data Science. 

As empresas preferem contratar Cientistas de Dados Plenos porque fornecem um valor tremendo a um salário razoável. Eles são mais experientes do que profissionais de nível Júnior, omitindo, assim, os caros erros do novato. Eles também não são tão caros quanto os Cientistas de Dados Seniores, embora ainda devam entregar modelos em produção. É um nível muito divertido, tendo ultrapassado o nível 1.0 e ainda tendo espaço para crescer para o nível 3.0.

O que eles fazem?
O Cientista de Dados Pleno domina a arte de construção de modelos analíticos e preditivos. Enquanto os Cientistas de Dados Seniores ou gerentes de negócios atribuem tarefas, o Cientista de Dados Pleno orgulha-se de criar produtos bem arquitetados. Ele evita falhas lógicas no modelo, duvida de sistemas com bom desempenho e orgulha-se de preparar os dados corretamente. O Cientista de Dados Pleno orienta os profissionais de nível Júnior e responde a perguntas de negócios para a gerência.

O que eles não fazem?
O Cientista de Dados Pleno não deve liderar equipes inteiras. Não é sua responsabilidade ter ideias para novos produtos, pois eles são gerados por colegas e gerentes mais experientes. Embora o Cientista de Dados Pleno conheça os detalhes dos produtos que eles criaram, não é esperado que eles conheçam a arquitetura geral de todos os produtos controlados por dados. O Cientista de Dados do nível 2.0 é especialista em Estatística e melhor em programação do que um Cientista de Dados do nível 1.0, mas se afasta da parte não divertida dos negócios no nível 3.0.

Um Cientista de Dados Pleno deve saber publicar seu código em produção (com algum apoio dos Engenheiros de Dados). 

O Cientista de Dados Pleno é avaliado pelo impacto que seus modelos geram. Ele(a) tem uma boa intuição sobre o funcionamento interno dos modelos estatísticos e como implementá-los e está no processo de entender melhor os negócios da empresa, mas não é esperado que forneça soluções para problemas de negócios ainda.

Cientista de Dados Sênior ou Principal – Nível Final 3.0
O Cientista de Dados Sênior é o membro mais experiente de uma equipe de Ciência de Dados. Tem mais de 5 anos de experiência e é versado em vários tipos de modelos de Ciência de Dados. Conhece as melhores práticas ao colocar modelos para trabalhar. Sabe escrever código computacionalmente eficiente e está sempre atento para encontrar projetos de negócios de alto impacto.

Além de suas impecáveis ​​habilidades de programação e profunda compreensão dos modelos científicos utilizados, ele(a) também entende firmemente os negócios em que sua empresa trabalha. Ele(a) tem um histórico de impactar a linha de base dos negócios com Data Science.

O Cientista de Dados Sênior precisa ter um entendimento muito bom do problema de negócios que está solucionando antes de escrever uma linha de código. Ou seja, eles precisam ter a capacidade de validar ideias antes da implementação. Essa abordagem aumenta o sucesso do projeto de Data Science. 

O que eles fazem?
O Cientista de Dados Sênior é responsável pela criação de projetos de Ciência de Dados de alto impacto. Em estreita coordenação com as partes interessadas, é responsável por liderar uma equipe potencialmente multifuncional no fornecimento da melhor solução para um determinado problema. Portanto, suas habilidades de liderança se desenvolveram desde os níveis 1.0 e 2.0. O Cientista de Dados Sênior atua como consultor técnico para Gerentes de Produto de diferentes departamentos. Com sua vasta experiência e habilidades nas principais categorias de Ciência de Dados, e se torna um ativo altamente valorizado para qualquer projeto.

O que eles não fazem?
Enquanto molda a discussão sobre as habilidades desejadas, não é responsabilidade do Cientista de Dados Sênior recrutar novos membros da equipe. Embora ele(a) entenda os negócios de sua empresa e sugira novos produtos impactantes, os gerentes de produto ainda são responsáveis ​​pela adoção no mercado. Ele(a) também lidera equipes, mas as decisões de progressão na carreira ainda são tomadas pelo líder da equipe.

O Cientista de Dados Sênior deve dirigir os projetos de acordo com as orientações do Chief Data Officer (CDO). Espera-se que essa pessoa obtenha as primeiras habilidades de liderança e, portanto, é importante que se comunique claramente, seja empático e tenha um bom olho para as pessoas.

O Cientista de Dados Sênior analisa porque os produtos de dados falham e, portanto, conduz novos projetos com sucesso. O profissional é um colaborador valioso das discussões sobre produtos e gosta de educar a empresa sobre Data Science. Com sua experiência no fornecimento de soluções impactantes de Data Science, é o ativo mais valioso do departamento de Ciência de Dados.

Conclusão
Navegar nos níveis de carreira em ciência de dados é divertido. Lembre-se dos seguintes tópicos principais:

Cientistas de Dados Júniores têm boas habilidades estatísticas e matemáticas.
Cientistas de Dados Plenos se destacam em colocar modelos em produção e dominam programação.
Cientistas de Dados Seniores sabem como criar valor comercial para suas soluções baseadas em dados.


