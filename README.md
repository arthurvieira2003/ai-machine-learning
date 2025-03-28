# Comitê de Classificadores para Aprendizagem Supervisionada

## Introdução

Este projeto implementa um comitê de classificadores usando aprendizagem supervisionada para classificar espécies de flores Iris. O objetivo é comparar o desempenho de diferentes algoritmos de classificação e combinar suas previsões para melhorar a precisão geral.

## Dataset Iris

### Descrição do Dataset

O conjunto de dados Iris é um dos conjuntos de dados mais conhecidos na área de aprendizado de máquina, introduzido pelo estatístico e biólogo Ronald Fisher em 1936. Contém medidas de 150 flores de íris pertencentes a três espécies diferentes:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

### Características das Instâncias e Atributos

O dataset contém 150 instâncias (50 de cada espécie) e 4 atributos numéricos:

1. **sepal length (cm)**: Comprimento da sépala em centímetros
2. **sepal width (cm)**: Largura da sépala em centímetros
3. **petal length (cm)**: Comprimento da pétala em centímetros
4. **petal width (cm)**: Largura da pétala em centímetros

Todas as variáveis são numéricas e contínuas, representando medidas físicas das flores.

### Variável Target (Rótulo da Classe)

A variável target é a espécie da flor, que pode ser:

- 0: Iris Setosa
- 1: Iris Versicolor
- 2: Iris Virginica

Esta é uma tarefa de classificação multiclasse, onde o objetivo é prever a espécie da flor com base em suas medidas.

## Algoritmos Utilizados

Este projeto utiliza os seguintes algoritmos de classificação:

1. **K-Nearest Neighbors (KNN)**: Classifica com base na similaridade (distância) entre os pontos de dados.
2. **Árvore de Decisão**: Utiliza uma estrutura de árvore para classificação baseada em regras.
3. **Naive Bayes**: Classificador probabilístico baseado no Teorema de Bayes.
4. **Support Vector Machine (SVM)**: Encontra o hiperplano que melhor separa as classes.
5. **Rede Neural (MLP)**: Utiliza uma rede neural de múltiplas camadas para classificação.

Além disso, esses classificadores são combinados em um **Comitê de Classificadores** usando um esquema de votação ponderada (soft voting).

## Métricas de Avaliação

Para avaliar e comparar o desempenho dos classificadores, utilizamos as seguintes métricas:

- **Acurácia**: Proporção de previsões corretas (taxa de acerto).
- **Taxa de Erro**: Proporção de previsões incorretas (1 - acurácia).
- **Matriz de Confusão**: Tabela que mostra as previsões corretas e incorretas para cada classe.
- **Precisão (Precision)**: Proporção de verdadeiros positivos entre todos os positivos previstos.
- **Sensibilidade/Recall**: Proporção de verdadeiros positivos identificados corretamente.
- **F1-Score**: Média harmônica entre precisão e recall.
- **Curva ROC (Receiver Operating Characteristic)**: Gráfico da taxa de verdadeiros positivos versus taxa de falsos positivos.
- **AUC (Area Under the Curve)**: Área sob a curva ROC, indicando a capacidade de discriminação do modelo.

## Resultados e Conclusões

Após executar o script principal (`comite_classificadores.py`), vários gráficos são gerados para visualizar o desempenho dos classificadores:

- **histogramas_atributos.png**: Distribuição dos valores de cada atributo para as diferentes classes.
- **matriz_correlacao.png**: Correlação entre os diferentes atributos.
- **graficos_dispersao.png**: Gráficos de dispersão mostrando a separação das classes no espaço de atributos.
- **comparacao_acuracias.png**: Comparação das acurácias de cada classificador.
- **comparacao_metricas.png**: Comparação de todas as métricas (precision, recall, F1-score, acurácia) entre os classificadores.
- **curvas_roc.png**: Curvas ROC para cada classe e cada classificador.

A análise desses resultados permitirá determinar qual algoritmo de aprendizado de máquina obteve o melhor desempenho na classificação das espécies de íris.

## Como Executar

Para executar o projeto, siga estas etapas:

1. Certifique-se de ter todas as dependências instaladas:

   ```
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Execute o script principal:

   ```
   python comite_classificadores.py
   ```

3. Analise os resultados nos arquivos gerados e no console.

## Conclusão

Este projeto demonstra a implementação e comparação de diferentes algoritmos de classificação usando o conjunto de dados Iris. O comitê de classificadores mostra como a combinação de diferentes modelos pode melhorar o desempenho geral da classificação.
