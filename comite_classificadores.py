#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comitê de Classificadores para Aprendizagem Supervisionada
"""

# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# Configuração para exibição de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Função para carregar e preparar os dados
def carregar_dados():
    """
    Carrega e prepara o conjunto de dados Iris
    """
    # Carregando o dataset Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Informações sobre o dataset
    print("Informações sobre o Dataset Iris:")
    print(f"Número de instâncias: {X.shape[0]}")
    print(f"Número de atributos: {X.shape[1]}")
    print(f"Classes: {data.target_names}")
    print(f"Atributos: {data.feature_names}")
    
    # Convertendo para DataFrame para facilitar a visualização
    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    return X, y, df, data.feature_names, data.target_names

# Função para explorar os dados
def explorar_dados(df):
    """
    Explora os dados através de estatísticas descritivas e visualizações
    """
    # Estatísticas descritivas
    print("\nEstatísticas Descritivas:")
    print(df.describe())
    
    # Verificar balanceamento das classes
    print("\nDistribuição das Classes:")
    print(df['species'].value_counts())
    
    # Criar visualizações
    plt.figure(figsize=(12, 8))
    
    # Histograma para cada atributo
    for i, col in enumerate(df.columns[:-2]):
        plt.subplot(2, 2, i+1)
        for species in df['species'].unique():
            plt.hist(df[df['species'] == species][col], bins=10, alpha=0.5, label=species)
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('histogramas_atributos.png')
    
    # Matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.iloc[:, :-2].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação dos Atributos')
    plt.tight_layout()
    plt.savefig('matriz_correlacao.png')
    
    # Gráfico de dispersão
    plt.figure(figsize=(12, 10))
    
    # Criar uma matriz de gráficos de dispersão para todas as combinações de atributos
    for i in range(4):
        for j in range(4):
            if i != j:
                plt.subplot(3, 4, i*3+j+1)
                for species, marker, color in zip(df['species'].unique(), ['o', 's', '^'], ['blue', 'green', 'red']):
                    temp = df[df['species'] == species]
                    plt.scatter(temp.iloc[:, i], temp.iloc[:, j], 
                               marker=marker, color=color, alpha=0.7, label=species)
                plt.xlabel(df.columns[i])
                plt.ylabel(df.columns[j])
                if i == 0 and j == 1:
                    plt.legend()
    
    plt.tight_layout()
    plt.savefig('graficos_dispersao.png')

# Função para treinar e avaliar os classificadores
def treinar_avaliar_modelos(X, y, nomes_atributos, nomes_classes):
    """
    Treina e avalia diferentes classificadores
    """
    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir os classificadores
    classifiers = [
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('Árvore de Decisão', DecisionTreeClassifier(random_state=42)),
        ('Naive Bayes', GaussianNB()),
        ('SVM', SVC(kernel='rbf', probability=True, random_state=42)),
        ('Rede Neural', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
    ]
    
    # Dicionários para armazenar resultados
    acuracias = {}
    reports = {}
    matrices = {}
    y_probs = {}  # Para a curva ROC
    
    # Treinar e avaliar cada classificador
    for nome, clf in classifiers:
        # Treinar o modelo
        clf.fit(X_train_scaled, y_train)
        
        # Fazer previsões
        y_pred = clf.predict(X_test_scaled)
        
        # Para curva ROC (é necessário obter probabilidades)
        if hasattr(clf, "predict_proba"):
            y_probs[nome] = clf.predict_proba(X_test_scaled)
        else:  # Para SVC sem probability=True
            y_probs[nome] = clf.decision_function(X_test_scaled)
            
        # Calcular a acurácia
        accuracy = accuracy_score(y_test, y_pred)
        acuracias[nome] = accuracy
        
        # Relatório de classificação
        report = classification_report(y_test, y_pred, target_names=nomes_classes, output_dict=True)
        reports[nome] = report
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        matrices[nome] = cm
        
        # Exibir resultados
        print(f"\nResultados para {nome}:")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Taxa de Erro: {1-accuracy:.4f}")
        print("\nMatriz de Confusão:")
        print(cm)
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=nomes_classes))
    
    # Criar comitê de classificadores (Voting)
    voting_clf = VotingClassifier(
        estimators=classifiers,
        voting='soft'  # 'soft' usa probabilidades, 'hard' usa previsões de classe
    )
    
    # Treinar o comitê
    voting_clf.fit(X_train_scaled, y_train)
    
    # Fazer previsões com o comitê
    y_pred_voting = voting_clf.predict(X_test_scaled)
    y_probs_voting = voting_clf.predict_proba(X_test_scaled)
    
    # Calcular métricas para o comitê
    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    voting_report = classification_report(y_test, y_pred_voting, target_names=nomes_classes, output_dict=True)
    voting_cm = confusion_matrix(y_test, y_pred_voting)
    
    # Adicionar resultados do comitê aos dicionários
    acuracias['Comitê'] = voting_accuracy
    reports['Comitê'] = voting_report
    matrices['Comitê'] = voting_cm
    y_probs['Comitê'] = y_probs_voting
    
    # Exibir resultados do comitê
    print("\nResultados para o Comitê de Classificadores:")
    print(f"Acurácia: {voting_accuracy:.4f}")
    print(f"Taxa de Erro: {1-voting_accuracy:.4f}")
    print("\nMatriz de Confusão:")
    print(voting_cm)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_voting, target_names=nomes_classes))
    
    return X_test_scaled, y_test, acuracias, reports, matrices, y_probs

# Função para plotar a curva ROC
def plotar_curva_roc(X_test, y_test, y_probs, nomes_classes):
    """
    Plota as curvas ROC para cada classificador
    """
    plt.figure(figsize=(10, 8))
    
    # Como o Iris tem três classes, precisamos calcular a curva ROC one-vs-rest para cada classe
    for i, classe in enumerate(nomes_classes):
        plt.subplot(2, 2, i+1)
        for nome, probs in y_probs.items():
            # Converter para formato binário one-vs-rest
            y_test_bin = (y_test == i).astype(int)
            
            # Diferentes formatos dependendo do classificador
            if probs.ndim == 1:  # Para SVC com decision_function
                # Precisamos ajustar para obter a probabilidade específica da classe
                y_score = probs.copy()
                fpr, tpr, _ = roc_curve(y_test_bin, y_score)
            else:  # Para classificadores com predict_proba
                fpr, tpr, _ = roc_curve(y_test_bin, probs[:, i])
            
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC para classe {classe}')
        plt.legend(loc="lower right")
    
    # Plotar a comparação geral
    plt.subplot(2, 2, 4)
    
    # Calcular a média ponderada da AUC para cada classificador
    for nome, probs in y_probs.items():
        if probs.ndim == 1:  # Para SVC com decision_function
            continue  # Pular SVC na média (não é trivial combinar decision_function)
        
        # Média micro das curvas ROC
        y_test_bin = np.eye(len(nomes_classes))[y_test]  # Converter para one-hot
        
        # Calcular micro-média da curva ROC
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC Média (micro)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('curvas_roc.png')

# Função para comparar os classificadores
def comparar_classificadores(acuracias, reports):
    """
    Compara os classificadores usando diferentes métricas
    """
    # Comparar acurácias
    plt.figure(figsize=(12, 6))
    plt.bar(acuracias.keys(), acuracias.values())
    plt.title('Comparação de Acurácias entre os Classificadores')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)
    for i, v in enumerate(acuracias.values()):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('comparacao_acuracias.png')
    
    # Comparar precision e recall médios
    precision = {name: report['weighted avg']['precision'] for name, report in reports.items()}
    recall = {name: report['weighted avg']['recall'] for name, report in reports.items()}
    f1 = {name: report['weighted avg']['f1-score'] for name, report in reports.items()}
    
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Acurácia': acuracias
    })
    
    plt.figure(figsize=(14, 8))
    metrics_df.plot(kind='bar', figsize=(14, 8))
    plt.title('Comparação de Métricas entre os Classificadores')
    plt.ylabel('Valor')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comparacao_metricas.png')
    
    # Exibir a tabela de métricas
    print("\nComparação de Métricas entre os Classificadores:")
    print(metrics_df)
    
    # Identificar o melhor classificador
    best_classifier = max(acuracias, key=acuracias.get)
    print(f"\nMelhor Classificador baseado na Acurácia: {best_classifier} ({acuracias[best_classifier]:.4f})")

# Função principal
def main():
    """
    Função principal para executar o pipeline completo
    """
    print("=" * 80)
    print("COMITÊ DE CLASSIFICADORES PARA APRENDIZAGEM SUPERVISIONADA")
    print("=" * 80)
    
    # Carregar e preparar os dados
    X, y, df, nomes_atributos, nomes_classes = carregar_dados()
    
    # Explorar os dados
    explorar_dados(df)
    
    # Treinar e avaliar os modelos
    X_test, y_test, acuracias, reports, matrices, y_probs = treinar_avaliar_modelos(X, y, nomes_atributos, nomes_classes)
    
    # Comparar os classificadores
    comparar_classificadores(acuracias, reports)
    
    # Plotar a curva ROC
    plotar_curva_roc(X_test, y_test, y_probs, nomes_classes)
    
    print("\nAnálise completa. Verifique os gráficos gerados para comparar visualmente os classificadores.")

if __name__ == "__main__":
    main() 