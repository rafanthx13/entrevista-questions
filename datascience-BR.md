# Data Science - BR

## Índice

## Links
+ https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo

## Conceituais

##### O que é DataScience?

Ciência de dados é a atividade de extrair informação a apartir da análise de dados (estruturados (ml) ou não estruturados (dl)).

Usase técnicas da matemática, estatística e computação (algoritmos, ml e dl) para fazer isso.

Data Science NÃO É...



...Inteligência Artificial.

...Estatística.

...Machine Learning.

...Big Data.

...Power BI.

...Algoritmos.


Data Science é a habilidade que você tem em extrair conhecimento de informações que podem ser analisadas.


##### O que é Big Data?

##### Qual é a diferença entre ‘data science’ e ‘big data’?

Big data significa ter um enorme volume de dados. O termo em sí significa isso mas pode engloba toda a parte de arquiteutra de uma empresa para suportar ter esse dados (como o haddop).

Big Data em si nâo traz valor algum sem a técnica de dataSicnece. ENtâo, Big Data é um objeto de análise para DataSicnec

##### Qual é a diferença entre um ‘data scientist’ e um ‘data analyst’?

A DataScience busca retirar informações aparitr de técnicas computaçiconais: Classificar, regressao e etc..

DataAnalises resolve problemas de negócios, utilizando mais a estatistica para resolver as coisas.

##### Quais são os recursos fundamentais que representam big data?

Agora que abordamos as definições, podemos passar para as perguntas mais específicas de uma entrevista sobre data science. Tenha em mente, porém, que você será obrigado a responder perguntas relacionadas a data scientist, data analyst e big data. A razão para isso acontecer é porque todas essas subcategorias estão interligadas entre si.

Existem cinco categorias que representam big data e são chamadas de ” 5 Vs “:

Valor;
Variedade;
Velocidade;
Veracidade;
Volume.
Todos esses termos correspondem ao Big Data de uma maneira ou de outra.

#####  What’s the difference between AI and ML?
AI and machine learning

(Source)

AI and ML are closely related, but these terms aren’t interchangeable. ML actually falls under the umbrella of AI. It demands that machines carry out tasks in the same way that humans do.

The current application of ML in AI is based around the idea that we should enable access to data so machines can observe and learn for themselves.

---
---
---

## Tecnicas 

###### O que é Overfitting, Underfitting e Generalization?

###### Quais técnicas utilizadas para tratamento de variáveis categóricas?

###### O que é ‘validação cruzada’?

cross-validation: É executar um modelo de ml de várias formas diferentes (mudando a ordem de entrada das rows do dataset)

Ele verifica como determinados resultados de análises estatísticas específicas serão medidos quando colocados em um conjunto independente de dados.

###### Qual é a diferença entre o aprendizado ‘supervisionado’ e ‘não supervisionado’?

Embora essa não seja uma das perguntas mais comuns das entrevistas, e, tenha mais a ver com machine learning do que com qualquer outra coisa, ela ainda assim pertence ao data science, portanto vale a pena saber a resposta.

Durante o aprendizado supervisionado, você infere uma função de uma parte rotulada de dados projetada para treinamento. Basicamente, a máquina aprenderia com os exemplos objetivos e concretos que você fornece.

Aprendizado não supervisionado refere-se a um método de treinamento de máquina que não usa respostas rotuladas – a máquina aprende por descrições dos dados de entrada.

###### Differentiate Machine Learning and Deep Learning?

ML é uma área de IA que trata de algorimtomos ou técnicas computacionais para máquinas/modelo/algoritmo aprender automaticamnete com os dados.

DL é um conjunto de ML que trata das redes neurias com várias camadas para fazer a mesma coias (porem mais complexa).

###### What makes Classification different from Regression

Both these concepts are an important aspect of supervised machine learning techniques. With Classification, the output is classified into different categories for making predictions. Whereas Regression models are usually used to find out the relationship between forecasting and variables. A key difference between classification and regression is that in the former the output variable is discrete and it is continuous in the latter.


####### How will you deal with missing data in a dataset?

One of the greatest challenges faced by a data scientist pertains to the problem of missing data. You can attribute the missing values in many ways including assigning a unique category, row deletion, substituting with mean/median/mode, employing algorithms that support the support missing values, and forecasting the missing value to name a few.

###### Explain what precision and recall are. How do they relate to the ROC curve?

https://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers.html/2

extrair tudo do link acima

The recall is alternatively called a true positive rate. It refers to the number of positives that have been claimed by your model compared to the number of positives that are available throughout the data.

Precision, which is alternatively called a positive predicted value, is based on prediction. It is a measurement of the number of accurate positives that the model has claimed as compared to the number of positives that the model has actually claimed.

######### What is the difference between type I vs type II error?

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


###### What’s regularization?

When you have underfitting or overfitting issues in a statistical model, you can use the regularization technique to resolve it. Regularization techniques like LASSO help penalize some model parameters if they are likely to lead to overfitting.

If the interviewer follows up with a question about other methods that can be used to avoid overfitting, you can mention cross-validation techniques such as k-folds cross-validation.

Another approach is to keep the model simple by taking into account fewer variables and parameters. Doing this helps remove some of the noise in the training data.

link: http://enhancedatascience.com/2017/07/04/machine-learning-explained-regularization/

###### What steps would you take to evaluate the effectiveness of your ML model?
You have to first split the data set into training and test sets. You also have the option of using a cross-validation technique to further segment the data set into a composite of training and test sets within the data.

Then you have to implement a choice selection of the performance metrics like the following:

Confusion matrix
Accuracy
Precision
Recall or sensitivity
Specificity
F1 score
For the most part, you can use measures such as accuracy, confusion matrix, or F1 score. However, it’ll be critical for you to demonstrate that you understand the nuances of how each model can be measured by choosing the right performance measure to match the problem.

https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b

###### What would you do if data in a data set were missing or corrupted?

Whenever data is missing or corrupted, you either replace it with another value or drop those rows and columns altogether. In Pandas, both isNull() and dropNA() are handy tools to find missing or corrupted data and drop those values. You can also use the fillna() method to fill the invalid values in a placeholder—for example, “0.”

disecar

https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/


##### Q18. While working on a data set, how do you select important variables? Explain your methods.

Answer: Following are the methods of variable selection you can use:

Remove the correlated variables prior to selecting important variables
Use linear regression and select variables based on p values
Use Forward Selection, Backward Selection, Stepwise Selection
Use Random Forest, Xgboost and plot variable importance chart
Use Lasso Regression
Measure information gain for the available set of features and select top n features accordingly.
 

##### Q19. What is the difference between covariance and correlation?

Answer: Correlation is the standardized form of covariance.

Covariances are difficult to compare. For example: if we calculate the covariances of salary ($) and age (years), we’ll get different covariances which can’t be compared because of having unequal scales. To combat such situation, we calculate correlation to get a value between -1 and 1, irrespective of their respective scale.

 

##### Q20. Is it possible capture the correlation between continuous and categorical variable? If yes, how?

Answer: Yes, we can use ANCOVA (analysis of covariance) technique to capture association between continuous and categorical variables.
------
------
-------
------

## Questoes cientificas de paper

https://www.springboard.com/blog/artificial-intelligence-questions/ 
aparitr de 38

##### What’s your favorite use case?

Just like research, you should be up to date on what’s going on in the industry. As such, if you’re asked about use cases, make sure that you have a few examples in mind that you can share. Whenever possible, bring up your personal experiences.

You can also share what’s happening in the industry. For example, if you’re interested in the use of AI in medical images, Health IT Analytics has some interesting use cases:

Detecting Fractures And Other Musculoskeletal Injuries
Aiding In The Diagnosis Neurological Diseases
Flagging Thoracic Complications And Conditions

##### What conferences are you hoping to attend this year? Any keynote speeches you’re hoping to catch?

Conferences are great places to network, attend workshops, learn, and grow. So if you’re planning to stick to a career in artificial intelligence, you should be going to some of these. For example, Deep Learning World has a great one every summer.

This year’s event in Las Vegas will feature keynote speakers like Dr. Dyann Daley (founder and CEO of Predict Align Prevent), Siddha Ganju (solutions architect at Nvidia), and Dr. Alex Glushkovsky (principal data scientist at BMO Financial Group, and others).

##### Do you have research experience in AI?

At present, a lot of work within the AI space is research-based. As a result, many organizations will be digging into your background to ascertain what kind of experience you have in this area. If you authored or co-authored research papers or have been supervised by industry leaders, make sure to share that information.

In fact, take it a step further and have a summary of your research experience along with your research papers ready to share with the interviewing panel.

However, if you don’t have any formal research experience, have an explanation ready. For example, you can talk about how your AI journey started as a weekend hobby and grew into so much more within a space of two or three years.


#### PERGUNTA MATADORA
!!!!!!!!!!!!!!!!!
**Repsonda todas essa pergunta sde forma mais curta para uma criança**
!!!!!!!!!!!!!!!!!

