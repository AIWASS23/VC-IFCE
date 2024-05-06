import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans

from google.colab import drive
drive.mount('/content/drive')
dados = pd.read_excel('/content/drive/MyDrive/banco_de_dados.xlsx', header = 1, usecols = range(3, 15))

dados_sem_faltantes = dados.dropna()

dados_sem_faltantes.to_excel('dados_faltantes.xlsx', index = False)

dados_sem_faltantes.loc[:, 'LC'] = dados_sem_faltantes['LC'].astype(float)
dados_sem_faltantes.loc[:, 'CE'] = dados_sem_faltantes['CE'].astype(float)
dados_sem_faltantes.loc[:, 'ROA'] = dados_sem_faltantes['ROA'].astype(float)

times = ['Athlético-PA', 'Atlético-GO', 'Atlético-MG', 'Bahia', 'Botafogo', 'Ceará', 'Corinthians',
         'Coritiba', 'Criciúma', 'Cruzeiro', 'Cuiabá', 'Flamengo', 'Fluminense', 'Fortaleza',
         'Goiás', 'Grêmio', 'Internacional', 'Palmeiras', 'Ponte Preta', 'Red Bull Bragantino', 'Santos',
         'São Paulo', 'Sport', 'Vasco']

medias_por_time_df = pd.DataFrame(columns=['TIME', 'AC', 'PC', 'LC', 'PNC', 'CE', 'RECEITA', 'LL', 'ATIVO TOTAL', 'ROA'])

for time in times:
    dados_time = dados_sem_faltantes[dados_sem_faltantes['TIME'] == time]
    media_time = dados_time[['AC', 'PC', 'LC', 'PNC', 'CE', 'RECEITA', 'LL', 'ATIVO TOTAL', 'ROA']].mean()
    media_time['TIME'] = time
    medias_por_time_df = pd.concat([medias_por_time_df, media_time.to_frame().transpose()], ignore_index = True)

medias_por_time_df.to_excel('dados_media.xlsx', index = False)

"""# Analise Exploratória"""

sns.pairplot(dados_sem_faltantes[['AC', 'PC', 'LC', 'PNC', 'CE', 'RECEITA', 'LL', 'ATIVO TOTAL', 'ROA']])
plt.show()

"""# Matriz de Correlação"""

correlacao = dados_sem_faltantes[['AC', 'PC', 'LC', 'PNC', 'CE', 'RECEITA', 'LL', 'ATIVO TOTAL', 'ROA']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title('Matriz de Correlação')
plt.show()

"""# Análise de Tendências

"""

time_escolhido = 'São Paulo'
dados_sem_faltantes['ANO'] = dados_sem_faltantes['ANO'].apply(np.floor)
plt.figure(figsize=(12, 8))

for indicador in ['AC', 'PC', 'LC', 'PNC', 'CE', 'RECEITA', 'LL', 'ATIVO TOTAL', 'ROA']:
    dados_time = dados_sem_faltantes[dados_sem_faltantes['TIME'] == time_escolhido]
    sns.lineplot(x='ANO', y=indicador, data=dados_time, label=f'{indicador} - {time_escolhido}')

plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.title('Tendências ao Longo do Tempo por Time')
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.show()

indicador_escolhido = 'RECEITA'
dados_sem_faltantes['ANO'] = dados_sem_faltantes['ANO'].apply(np.floor)

palette = sns.color_palette("Spectral", len(dados_sem_faltantes['TIME'].unique()))

plt.figure(figsize=(12, 8))

for i, time in enumerate(dados_sem_faltantes['TIME'].unique()):
    dados_time = dados_sem_faltantes[dados_sem_faltantes['TIME'] == time]
    sns.lineplot(x='ANO', y=indicador_escolhido, data=dados_time, label=f'{indicador_escolhido} - {time}', color = palette[i])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f'Tendências ao Longo do Tempo para o Indicador {indicador_escolhido}')
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.show()

"""# Analise Comparativa"""

indicador_escolhido = 'AC'
ranking = dados_sem_faltantes.groupby('TIME')[indicador_escolhido].mean().sort_values(ascending = False)

plt.figure(figsize = (12, 8))
ranking.plot(kind='bar', color = 'skyblue')
plt.title(f'Ranking de {indicador_escolhido} por Clube')
plt.xlabel('Clube')
plt.ylabel(f'{indicador_escolhido}')
plt.xticks(rotation = 45, ha = 'right')
plt.show()

"""# Modelagem Preditiva"""

X = dados_sem_faltantes[['AC', 'PC', 'LC', 'PNC', 'CE', 'RECEITA', 'LL', 'ATIVO TOTAL']]
y = dados_sem_faltantes['ROA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f'R-quadrado do conjunto de treinamento: {train_score:.2f}')
print(f'R-quadrado do conjunto de teste: {test_score:.2f}')

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Erro médio quadrático (MSE): {mse:.2f}')

"""# Análise de Componentes Principais"""

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x = X_pca[:, 0], y = X_pca[:, 1], hue = dados_sem_faltantes['TIME'])
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.title('Distribuição dos Clubes nos Componentes Principais')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

"""# Análise de Clusterização com K-means"""

kmeans = KMeans(n_clusters = 5, n_init = 10)
kmeans.fit(X)

dados_sem_faltantes['Cluster'] = kmeans.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dados_sem_faltantes['Cluster'], palette='viridis')
plt.title('Clusters de Clubes')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
