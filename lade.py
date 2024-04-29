# Importando Bibliotecas
import pandas as pd
import numpy as np
import graphviz
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz, plot_tree, export_text
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot
from sklearn.naive_bayes import  GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

# Criando conexão com o google drive
from google.colab import drive
drive.mount('/content/drive')

#Abrindo o banco de dados.
Integrado = pd.read_excel('/content/drive/MyDrive/Integrado_2018.2_a_2020_1_versão31_08_2022.xlsx')
Integrado

#Exibindo as Colunas, ou seja, seus nomes.
Integrado.columns

#Organizando a ordem das variáveis
Integrado=Integrado[['Semestre','Nota_Matemática', 'Pontuação', 'Reserva_vaga', 'Curso']]
Integrado

# convertendo o semestre da string - para podermos plotar alguns gráficos.
Integrado['Semestre']=Integrado['Semestre'].astype(str)
Integrado

#Verificando se existe dados faltantes e o tipo de variaveis
Integrado.info()

# Análise exploratória dos dados

#Estatística descritiva da pontuação final no exame de seleção por reserva de vaga.
grup_reserva=Integrado['Pontuação'].groupby(Integrado['Reserva_vaga'])
des=grup_reserva.describe()
des

mean_scores = grup_reserva.mean()
mean_scores


# Representação gráfica da pontuação final no exame de seleção por reserva de vaga.
plt.figure(figsize=(10, 6))
plt.plot(des['mean'])
plt.plot(des['mean'],'go')
plt.xlabel('Reserva de Vaga',fontsize=15)
plt.ylabel('Média no Exame de Seleção', fontsize=15)
plt.title('Média no Exame de Seleção Por Reserva de Vaga',fontsize=15 )
plt.show()

#Estatística descritiva da média em matemática no S1 por reserva de vaga.
grup_reserva_mat=Integrado['Nota_Matemática'].groupby(Integrado['Reserva_vaga'])
des_mat=grup_reserva_mat.describe()
des_mat


# Representação gráfica da média em matemática no S1 por reserva de vaga.
plt.figure(figsize=(10, 6))
plt.plot(des_mat['mean'])
plt.plot(des_mat['mean'],'go')
plt.xlabel('Reserva de Vaga',fontsize=15)
plt.ylabel('Média em Matemática no S1', fontsize=15)
plt.title('Média em Matemática no S1 Por Reserva de Vaga',fontsize=15 )
plt.show()

#Frequencia Absoluta da Reserva de Vaga
Reserva_Vaga = Integrado['Reserva_vaga'].value_counts()
Reserva_Vaga

# Representação Gráfica da Frequencia Absoluta da Reserva de Vaga
Reserva_Vaga.plot(kind='barh')

#Frequencia Relativa da Reserva de Vaga
Reserva_Vaga_Re=Reserva_Vaga / len(Integrado['Reserva_vaga'])
Reserva_Vaga_Re=Reserva_Vaga_Re*100
Reserva_Vaga_Re

# Representação Gráfica da Frequencia Relativa da Reserva de Vaga
Reserva_Vaga_Re.plot(kind='barh')

# Frequencia Absoluta dos Cursos
Curso = Integrado['Curso'].value_counts()
Curso

# Representação Gráfica da Frequencia Absoluta dos Cursos
Curso.plot(kind='barh')

#Frequencia Relativa dos Cursos
Curso_Re=Curso / len(Integrado['Curso'])
Curso_Re=Curso_Re*100
Curso_Re

# Representação Gráfica da Frequencia Relativa dos Cursos
Curso_Re.plot(kind='barh')

#Estatística descritiva da pontuação no exame de seleção  por curso.
grup_curso=Integrado['Pontuação'].groupby(Integrado['Curso'])
des_curso=grup_curso.describe()
des_curso

#Estatística descritiva da média em matemática por curso.
grup_curso=Integrado['Nota_Matemática'].groupby(Integrado['Curso'])
des_curso_Mat=grup_curso.describe()
des_curso_Mat

# Media em Matemática e Pontuação no exame de seleção semestre e curso.
grup_curso2=Integrado.groupby(['Semestre','Curso'])
des_sem_fis2=grup_curso2.mean()
des_sem_fis2

# Representação Gráfica da Media em Matemática, física e Pontuação no exame de seleção semestre e curso.
plt.figure(figsize=(30, 15))
sns.barplot(data=Integrado, x='Semestre',y='Nota_Matemática', hue='Curso')
plt.legend(loc = 2, bbox_to_anchor = (1,1))

#Estatística descritiva da média da pontuação no exame de seleção por semestre.
grup_curso=Integrado['Pontuação'].groupby(Integrado['Semestre'])
des_sem=grup_curso.describe()
des_sem

# Representação Gráfica da média da pontuação no exame de seleção por semestre.
plt.figure(figsize=(10, 6))
plt.plot(des_sem['mean'])
plt.plot(des_sem['mean'],'go')
plt.xlabel('Semestre',fontsize=15)
plt.ylabel('Média no Exame de Seleção', fontsize=15)
plt.title('Média no Exame de Seleção Por Semestre',fontsize=15 )
plt.show()

#Estatística descritiva da média em matemática no S1 por semestre.
grup_curso=Integrado['Nota_Matemática'].groupby(Integrado['Semestre'])
des_sem_mat=grup_curso.describe()
des_sem_mat

# Define o tamanho da figura do gráfico como 10 polegadas de largura por 6 polegadas de altura
plt.figure(figsize=(10, 6))

# Plota um gráfico de linha da média dos dados contidos na coluna 'mean' do DataFrame 'des_sem_mat'
plt.plot(des_sem_mat['mean'])

# Plota pontos verdes sobre a linha do gráfico de linha para representar os valores individuais da média
plt.plot(des_sem_mat['mean'], 'go')

# Define o rótulo do eixo x como 'Semestre' com tamanho de fonte 15
plt.xlabel('Semestre', fontsize=15)

# Define o rótulo do eixo y como 'Média em Matemática' com tamanho de fonte 15
plt.ylabel('Média em Matemática', fontsize=15)

# Define o título do gráfico como 'Média em Matemática Por Semestre' com tamanho de fonte 15
plt.title('Média em Matemática Por Semestre', fontsize=15)

# Mostra o gráfico
plt.show()

# Representação Gráfica da média em matemática no S1 por semestre.
plt.figure(figsize=(10, 6))
plt.plot(des_sem_mat['mean'])
plt.plot(des_sem_mat['mean'],'go')
plt.xlabel('Semestre',fontsize=15)
plt.ylabel('Média em Matemática', fontsize=15)
plt.title('Média em Matemática Por Semestre',fontsize=15 )
plt.show()

#Gráfico de dispersão Pontuação no Exame de Seleção x Média em Matemáitica no S1
pyplot.scatter(Integrado['Pontuação'], Integrado['Nota_Matemática'])
x=[0,1,2,3,4,5,6,7,8,9,10]
y=[6,6,6,6,6,6,6,6,6,6,6]
x1=[0,1,2,3,4,5,6,7,8,9,10]
y1=[6,6,6,6,6,6,6,6,6,6,6]
plt.plot(x,y, label='linear', marker = '>', color = 'r')
plt.plot(y1,x1, label='linear',  marker = 'v', color =  'r')
plt.xlabel('Pontuação no Exame de Seleção')
plt.ylabel('Média em Matemática no S1')
pyplot.title('Gráfico de Dispersão entre Pontuação no Exame de Seleção e Média em Matemática')
pyplot.show()

#Trasnformando reserva de vaga e curso em variável dummies
Integrado1=pd.get_dummies(Integrado,columns=['Reserva_vaga','Curso'],drop_first=True)

#Visualização do banco
Integrado1

#Criando a variável situação em matemática - categorica : Reprovado e Aprovado - Respectivamente com valores 0 e 1
Integrado1['Situação_Matemática']=pd.cut(Integrado1.Nota_Matemática, bins=[-1,5.9,10],labels=[0,1])
Integrado1

#Estatística Descritiva do banco Integrado1
Integrado1.describe()

#Separação do banco de dados em Regressores e Target
X_In=Integrado1.iloc[:,3:15]
y_In=Integrado1.iloc[:,15]

#Visualização dos Regressores
X_In

#Visualização dos Target
y_In

#Verificação da frequencia das classes
np.unique(Integrado1['Situação_Matemática'], return_counts = True)

#Verificação de dados faltantes
Integrado1.isnull().sum()

#Verificação do balanceamento das classes reprovado e aprovado (0 e 1)
sns.countplot(x =Integrado1['Situação_Matemática']);

#Balanceando as Classes

#Importando a bibloteca para balancear as classes

#Verificando o número de observações e o número de variáveis em X_In (temos 468 observações e 13 variáveis)
X_In.shape

#Produzindo dados sinteticos para balancear as classes, aumentaremos o número de observações na classe 0, a minoritária.
smote = SMOTE(sampling_strategy='minority')
X_over, y_over = smote.fit_resample(X_In, y_In)

#Verificando como ficou o novo comjunto de regressores X_In, que se transformou em X_over com 714 observações
X_over.shape

#Verificando o numero de observações na variável target y_In, antes do balanceamento.
np.unique(y_In, return_counts=True)

#Verificando o numero de observações na variável target y_In, agora chamada de y_over, depois do balanceamento.
np.unique(y_over, return_counts=True)

#Visualização gráfica das classes depois do balanceamneto.
sns.countplot(x =y_over)

#Divisão do conjunto em treino e teste
X_treinamento_over, X_teste_over, y_treinamento_over, y_teste_over = train_test_split(X_over, y_over, test_size=0.15, random_state=0)
X_treinamento_over.shape, X_teste_over.shape

# Usando Regressão Logística

#Importanto a biblioteca para realizar a regressão logistica.

#Treinando o algoritmo
logistic= LogisticRegression(random_state = 1)
logistic.fit(X_treinamento_over, y_treinamento_over)

#Matriz de confusão
ConfusionMatrixDisplay.from_estimator(logistic,X_teste_over, y_teste_over)
plt.show()

#Calculando a Acuracia balanceada
clfs = [logistic]
indices = ['Logística']

bal_acc_results_list = []

for clf in clfs:
  train_acc = balanced_accuracy_score(y_treinamento_over, clf.predict(X_treinamento_over))
  test_acc  = balanced_accuracy_score(y_teste_over, clf.predict(X_teste_over))

  bal_acc_results_list.append({'Treino': train_acc, 'Teste': test_acc})

df_bal_acc_results = pd.DataFrame(bal_acc_results_list)

#Criando gráfico
df_bal_acc_results.insert(2, "Criterio", indices, True)
df_bal_acc_results = df_bal_acc_results.set_index('Criterio')

#Criando gráfico
ax = df_bal_acc_results.plot.bar(figsize=[8,6], legend=True, fontsize=18, rot=45, grid=True,
                            yticks=[0.0,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8, 0.85, 0.9, 0.95, 1.0],
                            ylim=[0.0, 1.0])
ax.set_title(label='Regressão Logística ', fontsize=25)
ax.set_xlabel(xlabel='Critério', fontsize=16)
plt.show()

#Criando a lista de previsões com os regressores X_over e quardando as observações
observado2 =y_over
previsto2 = logistic.predict(X_over)

#Calculando precision, recall, f1-score
print(metrics.classification_report(observado2, previsto2))
print(metrics.confusion_matrix(observado2, previsto2))

#Divisão em Treino e Teste
#X_treinamento_over, X_teste_over, y_treinamento_over, y_teste_over = train_test_split(X_over, y_over, test_size=0.15, random_state=0)
#X_treinamento_over.shape, X_teste_over.shape

#
y_scores=cross_val_predict(logistic, X_treinamento_over,y_treinamento_over, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_treinamento_over,y_scores)

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)

plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
plt.show()

plt.ylabel('Precisions')
plt.xlabel('Recalls')
plt.plot(precisions, recalls)
plt.show()

fpr, tpr,thresholds=roc_curve(y_treinamento_over,y_scores)
roc_auc = auc(fpr, tpr)
#label='ROC curve (area = %0.2f)' % roc_auc
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

# Neste [link](https://docs.google.com/spreadsheets/d/10FdAsAUUugwSAVMxgLl2Dqs_YO81SNVSrmCmvG01oo8/edit#gid=1592930993) preparei uma planilha com um exemplo para você entender melhor os passos citados.

plot_roc_curve(fpr, tpr)
plt.show()

# Usando Naive Bayes

naive = GaussianNB()
naive.fit(X_treinamento_over, y_treinamento_over)
previsoes2 = naive.predict(X_teste_over)
previsoes2

clfs3 = [naive]
indices = ['Naive Bayes']

bal_acc_results_list = []

for clf in clfs3:
  train_acc = balanced_accuracy_score(y_treinamento_over, clf.predict(X_treinamento_over))
  test_acc  = balanced_accuracy_score(y_teste_over, clf.predict(X_teste_over))

  bal_acc_results_list.append({'Treino': train_acc, 'Teste': test_acc})

df_bal_acc_results = pd.DataFrame(bal_acc_results_list)

df_bal_acc_results.insert(2, "Critério", indices, True)
df_bal_acc_results = df_bal_acc_results.set_index('Critério')

ax = df_bal_acc_results.plot.bar(figsize=[8,6], legend=True, fontsize=18, rot=45, grid=True,
                            yticks=[0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8, 0.85, 0.9, 0.95, 1.0],
                            ylim=[0.0, 1.0])
ax.set_title(label='Naive Bayes', fontsize=25)
ax.set_xlabel(xlabel='Critério', fontsize=16)
plt.show()

y_scores=cross_val_predict(naive, X_treinamento_over,y_treinamento_over, cv=3)

observado2 =y_over
previsto2 = naive.predict(X_over)

print(metrics.classification_report(observado2, previsto2))
print(metrics.confusion_matrix(observado2, previsto2))

precisions, recalls, thresholds = precision_recall_curve(y_treinamento_over,y_scores)

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)

plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
plt.show()

plt.ylabel('Precisions')
plt.xlabel('Recalls')
plt.plot(precisions, recalls)
plt.show()

fpr, tpr,thresholds=roc_curve(y_treinamento_over,y_scores)
roc_auc = auc(fpr, tpr)
#label='ROC curve (area = %0.2f)' % roc_auc
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

plot_roc_curve(fpr, tpr)
plt.show()