import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Carregar o modelo treinado
model = joblib.load("modelo_iris_rf.pkl")

# Carregar o dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Calcular acurácia do modelo
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

st.title("Classificação de Espécies de Iris")
st.write("Insira os valores para prever a espécie de uma flor Iris ou faça o upload de um arquivo CSV.")

# Sidebar para entrada de dados
sepal_length = st.sidebar.slider("Comprimento da Sépala (cm)", float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.sidebar.slider("Largura da Sépala (cm)", float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.sidebar.slider("Comprimento da Pétala (cm)", float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.sidebar.slider("Largura da Pétala (cm)", float(X[:, 3].min()), float(X[:, 3].max()))

# Botão para realizar a classificação
if st.button("Classificar uma amostra"):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    df = pd.DataFrame(data, columns=iris.feature_names)
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    st.write(f"**Espécie Prevista**: {iris.target_names[prediction[0]]}")
    st.write("**Probabilidades:**")
    for i, species in enumerate(iris.target_names):
        st.write(f"{species}: {prediction_proba[0][i] * 100:.2f}%")

# Upload de arquivo CSV para classificação em lote
st.subheader("Classificação em Lote com Arquivo CSV")
uploaded_file = st.file_uploader("Faça upload de um arquivo CSV com as características das flores", type="csv")

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    st.write("Amostras carregadas:")
    st.write(df_csv.head())
    # Fazer a previsão para cada linha do arquivo
    predictions_csv = model.predict(df_csv)
    df_csv['Previsao'] = [iris.target_names[i] for i in predictions_csv]
    st.write("Resultados das previsões:")
    st.write(df_csv)

# Visualização dos dados
st.subheader("Visualização dos Dados")
sns.set(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(X[:, 0], kde=True, ax=ax[0], color='blue').set(title='Distribuição do Comprimento da Sépala')
sns.histplot(X[:, 2], kde=True, ax=ax[1], color='green').set(title='Distribuição do Comprimento da Pétala')
st.pyplot(fig)

# Exibir a acurácia do modelo
st.subheader("Métricas de Avaliação do Modelo")
st.write(f"Acurácia do modelo RandomForest no conjunto de dados Iris: {accuracy * 100:.2f}%")

