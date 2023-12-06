import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando a biblioteca que tem o teste ks_2amp (Kolmogorov-Smirnov) que
# compara os retorno de duas amostras utilizado para testar a hipótese nula
# de que as amostras independentes são provenientes da mesma distribuição

from scipy.stats import ks_2samp

df = pd.read_csv('dados.csv', sep=';', decimal=',', header=None)

df.columns = ['Data', 'Valor']

valor_inicial = df['Valor'][0]


# Convertendo a primeira coluna para o formato de data
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%y')

# Renomeando as colunas
df.columns = ['Data', 'Valor']

# Calculando a divisão do valor pelo valor anterior
df['Razao'] = df['Valor'] / df['Valor'].shift(1)

# Calculando o logaritmo natural
df['Retornos'] = np.log(df['Razao'])

# Criando uma cópia do DataFrame original com a coluna Retorno apenas
df_sorted = df[['Retornos']].copy()

# Removendo a primeira linha (índice 0) que contem NA
df_sorted.drop(df_sorted.index[0], inplace=True)

# Ordenando o DataFrame copiado pelos retornos
df_sorted = df_sorted.sort_values(by='Retornos')

# Redefinindo o índice
df_sorted.reset_index(drop=True, inplace=True)

# Variáveis com o retorno mínimo e máximo atribuídos à série
retorno_minimo = -0.10
retorno_maximo = 0.05

# Criando um novo DataFrame com a linha a ser adicionada
nova_linha_inicial = pd.DataFrame({'Retornos': [retorno_minimo]})
nova_linha_final = pd.DataFrame({'Retornos': [retorno_maximo]})
df_sorted = pd.concat([nova_linha_inicial, df_sorted, nova_linha_final]).reset_index(drop=True)

# Número total de linhas no DataFrame
total_linhas = len(df_sorted)-1

# Criando a nova coluna 'F_Chapeu'
df_sorted['F_Chapeu'] = (df_sorted.index) / total_linhas

# Número de períodos para os quais vou simular os retornos aleatórios
num_periodo = 252

# Estabelecendo o número de amostras
num_amostras = 100

# Criando a matriz de números aleatórios num_periodo linhas e 1000 colunas
dados_aleatorios = np.random.rand(num_periodo, num_amostras)

# Extraindo os valores das colunas para interpolação
x = df_sorted['F_Chapeu'].to_numpy()


y = df_sorted['Retornos'].to_numpy()


# Array para armazenar os resultados da interpolação
resultados_interpolados = np.zeros(dados_aleatorios.shape)

# Realizando a interpolação para cada elemento em dados_aleatorios
for i in range(dados_aleatorios.shape[0]):
    for j in range(dados_aleatorios.shape[1]):
        resultados_interpolados[i, j] = np.interp(dados_aleatorios[i, j], x, y)

# Criando a nova matriz
nova_matriz = np.zeros((num_periodo+1, num_amostras))

# Preenchendo a primeira linha com o valor inicial
nova_matriz[0, :] = valor_inicial

# Calculando os valores para as linhas subsequentes
for i in range(1, num_periodo+1):
    for j in range(num_amostras):
        nova_matriz[i, j] = nova_matriz[i-1, j] * np.exp(resultados_interpolados[i-1, j])

# Convertendo a matriz NumPy em um DataFrame
df_nova_matriz = pd.DataFrame(nova_matriz, columns=[f'Amostra_{i}' for i in range(num_amostras)])

# Criando um novo DataFrame com a coluna 'Data'
df_data = pd.DataFrame({'Data': df['Data'][:num_periodo+1]})

# Concatenando df_nova_matriz com df_data
df_nova_matriz_com_data = pd.concat([df_data, df_nova_matriz], axis=1)

# Convertendo cada valor numérico para string, substituindo ponto por vírgula
df_nova_matriz_com_data = df_nova_matriz_com_data.map(lambda x: str(x).replace('.', ','))

# Salvando o DataFrame como um arquivo CSV com ponto e vírgula como separador
df_nova_matriz_com_data.to_csv('novos_dados.csv', sep=';', index=False)

# Criando uma função para calcular retornos logarítmicos que será usada múltiplas vezes,
# assumindo que a primeira coluna é a data e as demais são valores
# Calcula os retornos logarítmicos de todas as colunas de valor

def calcular_retornos_logaritmicos(df):

    retornos = pd.DataFrame()
    for coluna in df.columns[1:]:  # Ignora a primeira coluna (data)
        # Ignora a primeira linha da coluna atual
        valores_deslocados = df[coluna].shift(1)
        valores_atuais = df[coluna]

        # Calcula os retornos logarítmicos, começando da segunda linha
        retornos_log = np.log(valores_atuais[1:] / valores_deslocados[1:])
        
        # Adiciona os retornos logarítmicos ao DataFrame de retornos
        retornos[coluna + '_Retornos_Log'] = retornos_log

    # Remove quaisquer linhas com NaN que possam ter sido geradas
    return retornos.dropna()

# Calculando os retornos para os dados do arquivo original
df = pd.read_csv('dados.csv', sep=';', decimal=',')
df.columns = ['Data', 'Valor']
retornos_originais = calcular_retornos_logaritmicos(df)

# Salvando o DataFrame como um arquivo CSV com ponto e vírgula como separador
retornos_originais.to_csv('retornos_originais.csv', sep=';', index=False)
retornos_originais = retornos_originais.drop(retornos_originais.index[0])

# Calculando os retornos para os dados do arquivo original
df_novo = pd.read_csv('novos_dados.csv', sep=';', decimal=',')
retornos_novos = calcular_retornos_logaritmicos(df_novo)

# Salvando o DataFrame como um arquivo CSV com ponto e vírgula como separador
retornos_novos.to_csv('retornos_novos.csv', sep=';', index=False)
retornos_novos = retornos_novos.drop(retornos_novos.index[0])


# Inicializando um DataFrame para armazenar os resultados
resultados_ks = pd.DataFrame(columns=['Amostra', 'Estatistica_KS', 'Valor_p'])

i = 1
# Iterando sobre as colunas e realizando o teste KS
for coluna in retornos_novos.columns:  # Ignorando a primeira coluna (assumindo que é 'Data')
    
    # Realizando o teste KS
    resultado_ks = ks_2samp(retornos_originais.iloc[1:, 0], retornos_novos[coluna].iloc[1:])
    print("Coluna: ", coluna, "Estatística KS :", resultado_ks.statistic, "p-Value ", resultado_ks.pvalue)
    
    # Adicionando os resultados ao DataFrame
    resultados_ks.loc[i] = ['Amostra' + str(i), resultado_ks.statistic, resultado_ks.pvalue]
    i += 1
    
# Salvando os resultados em um arquivo CSV
resultados_ks.to_csv('resultados_ks.csv', index=False)

# Criando o histograma para a coluna 'Valor_p' com intervalos de 0.05
plt.figure(figsize=(10, 6))
plt.hist(resultados_ks['Valor_p'], bins=np.arange(0, 1.05, 0.05), edgecolor='black')
plt.title('Histograma dos Valores p')
plt.xlabel('Valor_p')
plt.ylabel('Frequência')
plt.xticks(np.arange(0, 1.05, 0.05))
plt.grid(axis='y', alpha=0.75)

# Salvando o histograma em um arquivo PNG
plt.savefig('./histograma_valores_p.png')

# Criando o histograma para a coluna 'Estatistica_KS' com intervalos de 0.05
plt.figure(figsize=(10, 6))
plt.hist(resultados_ks['Estatistica_KS'], bins=np.arange(0, 1.05, 0.05), edgecolor='black')
plt.title('Histograma das Estatisticas KS')
plt.xlabel('Estatistica_KS')
plt.ylabel('Frequência')
plt.xticks(np.arange(0, 1.05, 0.05))
plt.grid(axis='y', alpha=0.75)

# Salvando o histograma em um arquivo PNG
plt.savefig('./histograma_estatistica_KS.png')

