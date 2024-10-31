import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import joblib

# Carregar o conjunto de dados Iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# Carregar modelos
model_rf = joblib.load("modelo_rf.pkl")
model_knn = joblib.load("modelo_knn.pkl")
model_svm = joblib.load("modelo_svm.pkl")
modelos = {"Random Forest": model_rf, "K-Nearest Neighbors": model_knn, "Support Vector Machine": model_svm}

# Título e menu de navegação
st.title("Dashboard de Classificação de Iris")
st.sidebar.title("Navegação")
selecao = st.sidebar.radio("Escolha a seção", ["Visão Geral", "Classificação de Espécies", "Gráficos Interativos", "Estatísticas Descritivas"])

# Seção de Visão Geral
if selecao == "Visão Geral":
    st.header("Visão Geral")
    st.write("Este dashboard permite explorar diferentes aspectos do conjunto de dados Iris.")
    st.write("Escolha um modelo de classificação, visualize gráficos interativos e veja estatísticas descritivas das espécies.")
    
    # Classificação em lote com upload de CSV
    st.subheader("Classificação em Lote com Arquivo CSV")
    uploaded_file = st.file_uploader("Faça upload de um arquivo CSV com as características das flores", type="csv")
    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        st.write("Amostras carregadas:")
        st.write(df_csv.head())
        
        # Fazer a previsão para cada linha do arquivo
        modelo_selecionado = st.selectbox("Escolha o modelo para classificação em lote", list(modelos.keys()))
        model = modelos[modelo_selecionado]
        predictions_csv = model.predict(df_csv)
        df_csv['Previsao'] = [iris.target_names[i] for i in predictions_csv]
        st.write("Resultados das previsões:")
        st.write(df_csv)

# Seção de Classificação de Espécies
elif selecao == "Classificação de Espécies":
    st.header("Classificação de Espécies de Iris")
    st.write("Escolha um modelo de classificação, visualize dados e faça previsões individuais ou em lote.")
    
    # Seleção do modelo e entrada de dados para classificação
    modelo_selecionado = st.selectbox("Escolha o modelo de classificação", list(modelos.keys()))
    model = modelos[modelo_selecionado]

    # Sidebar para entrada de dados
    sepal_length = st.sidebar.slider("Comprimento da Sépala (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()))
    sepal_width = st.sidebar.slider("Largura da Sépala (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()))
    petal_length = st.sidebar.slider("Comprimento da Pétala (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()))
    petal_width = st.sidebar.slider("Largura da Pétala (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()))

    if st.button("Classificar uma amostra"):
        # Criar DataFrame com os valores inseridos
        data = [[sepal_length, sepal_width, petal_length, petal_width]]
        df = pd.DataFrame(data, columns=iris.feature_names)
        
        # Fazer a previsão
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df) if hasattr(model, "predict_proba") else [[1.0, 0.0, 0.0]]  # Verifica se o modelo suporta probabilidade
        
        # Exibir a espécie prevista
        st.write(f"**Modelo Selecionado**: {modelo_selecionado}")
        st.write(f"**Espécie Prevista**: {iris.target_names[prediction[0]]}")
        
        # Exibir as probabilidades de cada classe
        st.write("**Probabilidades:**")
        for i, species in enumerate(iris.target_names):
            st.write(f"{species}: {prediction_proba[0][i] * 100:.2f}%")

# Seção de Gráficos Interativos
elif selecao == "Gráficos Interativos":
    st.header("Gráficos Interativos")
    plot_tipo = st.selectbox("Escolha o tipo de gráfico", ["Scatter Plot", "Gráfico de Barras", "Histograma", "Boxplot"])

    if plot_tipo == "Scatter Plot":
        # Scatter plot entre comprimento e largura das pétalas
        st.write("Scatter Plot: Comprimento vs. Largura da Pétala")
        fig, ax = plt.subplots()
        sns.scatterplot(data=iris_df, x="petal length (cm)", y="petal width (cm)", hue="species", ax=ax)
        st.pyplot(fig)

    elif plot_tipo == "Gráfico de Barras":
        # Gráfico de barras da quantidade de cada espécie
        st.write("Gráfico de Barras: Quantidade de cada Espécie")
        fig, ax = plt.subplots()
        iris_df['species'].value_counts().plot(kind="bar", color=['blue', 'green', 'red'], ax=ax)
        ax.set_ylabel("Quantidade")
        ax.set_title("Distribuição das Espécies")
        st.pyplot(fig)
    
    elif plot_tipo == "Histograma":
        # Histograma do comprimento da sépala
        st.write("Histograma: Comprimento da Sépala")
        fig, ax = plt.subplots()
        sns.histplot(iris_df['sepal length (cm)'], kde=True, ax=ax, color='purple')
        ax.set_title("Distribuição do Comprimento da Sépala")
        st.pyplot(fig)
    
    elif plot_tipo == "Boxplot":
        # Boxplot do comprimento da sépala por espécie
        st.write("Boxplot: Comprimento da Sépala por Espécie")
        fig, ax = plt.subplots()
        sns.boxplot(data=iris_df, x="species", y="sepal length (cm)", palette="Set2", ax=ax)
        ax.set_title("Comprimento da Sépala por Espécie")
        st.pyplot(fig)

# Seção de Estatísticas Descritivas
elif selecao == "Estatísticas Descritivas":
    st.header("Estatísticas Descritivas")
    especie_selecionada = st.selectbox("Escolha a espécie para visualização", ["setosa", "versicolor", "virginica"])

    # Filtrar o dataset para a espécie selecionada
    especie_df = iris_df[iris_df['species'] == especie_selecionada]

    # Cálculo de estatísticas descritivas
    media_sepal_length = especie_df['sepal length (cm)'].mean()
    media_petal_length = especie_df['petal length (cm)'].mean()
    desvio_sepal_length = especie_df['sepal length (cm)'].std()
    desvio_petal_length = especie_df['petal length (cm)'].std()

    st.subheader(f"Métricas da Espécie: {especie_selecionada.capitalize()}")
    st.write(f"Média do Comprimento da Sépala: {media_sepal_length:.2f} cm")
    st.write(f"Desvio Padrão do Comprimento da Sépala: {desvio_sepal_length:.2f} cm")
    st.write(f"Média do Comprimento da Pétala: {media_petal_length:.2f} cm")
    st.write(f"Desvio Padrão do Comprimento da Pétala: {desvio_petal_length:.2f} cm")

    st.write("Espécies disponíveis: Setosa, Versicolor, Virginica.")
    st.write("A navegação acima permite comparar e explorar diferentes espécies de maneira interativa.")

