# Machine Learning For Heart Attack Prediction

## PT-BR

### Autora

Jéssica Raissa Pessoa - Bacharelanda em Ciência da Computação (FPB) e mestra em Comunicação e Culturas Midiáticas (UFPB)

- Link do meu perfil no Github: [https://github.com/jessicaraissapessoa](https://github.com/jessicaraissapessoa)
- Link do notebook desse trabalho no Kaggle: [https://www.kaggle.com/code/jssicaraissa/heart-attack-prediction-with-machine-learning](https://www.kaggle.com/code/jssicaraissa/heart-attack-prediction-with-machine-learning)
- Link do repositório Github desse trabalho: [https://github.com/jessicaraissapessoa/MachineLearningForHeartPredictionAttack](https://github.com/jessicaraissapessoa/MachineLearningForHeartPredictionAttack)

### Descrição

Esse trabalho é um estudo aplicando modelos de machine learning para previsão de valores. O dataset utilizado apresenta dados de exames de coração de pacientes, com objetivo de traçar fatores que possam estar relacionado à uma maior ou menor chance de ataque cardíaco.

Link do dataset: [https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
Visto que a variável alvo possui como valores as categorias 0 (menor chance) e 1 (maior chance), trata-se de um contexto de aplicação de modelos com algoritmo de classificação. Foi aplicado aprendizado supervisionado.

Modelos de machine learning aplicados:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Random Forest Classifier
- Decision Tree Classifier
- Multi-Layer Perceptron Classifier (MLP Classifier)

### Objetivo

- Aplicar, na prática, conhecimentos de machine learning
- Analisar dados, obter insights e identificar padrões e relações entre valores obtidos em exames de coração e a probabilidade de ataque cardíaco

### Variáveis preditoras (utilizadas para predição - x)

A partir do mapa de calor, foi determinado que as seguintes variáveis apresentavam correlação mais interessante, sendo, portanto, as selecionadas:

- cp: tipo da dor no peito (chest pain - cp)
  - 1: angina típica
  - 2: angina atípica
  - 3: dor não anginosa
  - 4: assintomático
- thalachh: frequência cardíaca máxima alcançada
- slp: declive
- oldpeak: pico anterior
- exng: angina induzida por exercício
  - 1: sim
  - 2: não

### Variável alvo (a ser prevista - y)

- target: chance de ataque cardíaco
  - 0: menor chance
  - 1: maior chance

## EN-US

### Author

Jéssica Raissa Pessoa - Bachelor's degree student in Computer Science (FPB) and Master's in Communication and Media Cultures (UFPB)

- Link to my Github profile: [https://github.com/jessicaraissapessoa](https://github.com/jessicaraissapessoa)
- Link to the notebook of this work on Kaggle: [https://www.kaggle.com/code/jssicaraissa/heart-attack-prediction-with-machine-learning](https://www.kaggle.com/code/jssicaraissa/heart-attack-prediction-with-machine-learning)
- Link to the Github repository of this work: [https://github.com/jessicaraissapessoa/MachineLearningForHeartPredictionAttack](https://github.com/jessicaraissapessoa/MachineLearningForHeartPredictionAttack)

### Description

This work is a study applying machine learning models for value prediction. The dataset used presents heart exam data from patients, aiming to determine factors that may be related to a higher or lower chance of a heart attack.

Link to the dataset: [https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
Given that the target variable has values of 0 (lower chance) and 1 (higher chance), this is a context for applying models with a classification algorithm. Supervised learning was applied.

Machine learning models applied:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Random Forest Classifier
- Decision Tree Classifier
- Multi-Layer Perceptron Classifier (MLP Classifier)

### Objective

- Practically apply machine learning knowledge
- Analyze data, gain insights, and identify patterns and relationships between heart exam results and the probability of a heart attack

### Predictor variables (used for prediction - x)

From the heatmap, it was determined that the following variables had the most interesting correlation and were therefore selected:

- cp: type of chest pain
  - 1: typical angina
  - 2: atypical angina
  - 3: non-anginal pain
  - 4: asymptomatic
- thalachh: maximum heart rate achieved
- slp: slope
- oldpeak: previous peak
- exng: exercise-induced angina
  - 1: yes
  - 2: no

### Target variable (to be predicted - y)

- target: chance of heart attack
  - 0: lower chance
  - 1: higher chance
