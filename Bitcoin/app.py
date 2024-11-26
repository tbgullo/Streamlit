import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

def main():
    st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
    # Carregar o modelo treinado
    model = joblib.load('Bitcoin/model.pkl')

    # Sidebar navigation
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Selecione a página:", ["Panorama Geral", "Deduzir Preço"])
    
    if page == "Panorama Geral":
        panorama_geral()
    elif page == "Deduzir Preço":
        deduzir_preco(model)

def panorama_geral():
    st.title("Panorama Geral do Bitcoin")
    
    # Carregar dados do arquivo CSV (BTC-USD.csv)
    data = pd.read_csv('Bitcoin/BTC-USD.csv', parse_dates=['Date'])
    
    # Filtrando apenas a coluna 'Data' e 'Close' (preço de fechamento)
    data = data[['Date', 'Close']]
    data.set_index('Date', inplace=True)

    # Exibir as primeiras linhas dos dados
    st.write(data.head())
    
    # Gráfico de seleção
    grafico = st.selectbox("Escolha o tipo de gráfico:", ["Linha", "Barra", "Histograma"])
    
    if grafico == "Linha":
        st.line_chart(data['Close'])
    elif grafico == "Barra":
        st.bar_chart(data['Close'])
    elif grafico == "Histograma":
        fig, ax = plt.subplots()
        ax.hist(data['Close'], bins=20, color='blue', edgecolor='black')
        ax.set_title('Distribuição de Preços de Fechamento')
        ax.set_xlabel('Preço de Fechamento')
        ax.set_ylabel('Frequência')
        st.pyplot(fig)

def deduzir_preco(model):
    st.title("Deduzir Preço do Bitcoin")
    
    st.write("Ajuste as métricas abaixo para prever o preço do Bitcoin.")
    
    # Sliders para métricas
    preco_abertura = st.slider("Preço de Abertura:", min_value=20000, max_value=50000, value=30000, step=500)
    maior_preco = st.slider("Maior Preço:", min_value=20000, max_value=50000, value=40000, step=500)
    menor_preco = st.slider("Menor Preço:", min_value=20000, max_value=50000, value=25000, step=500)

    # Prevendo preço (exemplo fictício, substitua pelo seu modelo)
    preco_previsto = model.predict([preco_abertura, maior_preco , menor_preco])

    st.subheader("Preço Previsto")
    st.write(f"R$ {preco_previsto:,.2f}")

if __name__ == "__main__":
    main()
