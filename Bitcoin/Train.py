import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

base_bitcoin = pd.read_csv('BTC-USD.csv')

# Colunas Adj Close e Volume não iriam impactar positivamente
base_bitcoin = base_bitcoin.drop(['Adj Close', 'Volume'], axis=1)

# Converter Date para id
base_bitcoin['Date'] = pd.to_datetime(base_bitcoin['Date'])
base_bitcoin['Date'] = (base_bitcoin['Date'] - base_bitcoin['Date'].min()).dt.days

# Definir X (variáveis independentes) e y (variável dependente)
X_bitcoin = base_bitcoin.iloc[: , 0:4]
y_bitcoin = base_bitcoin.iloc[: , 4]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_bitcoin, y_bitcoin, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

previsoes = model.predict(X_test)

# Avaliação do modelo de regressão
mse = mean_squared_error(y_test, previsoes)
mae = mean_absolute_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R² Score: {r2}')


import pickle

pickle.dump(model, open('model.pkl' , "wb"))