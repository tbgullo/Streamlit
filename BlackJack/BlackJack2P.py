import streamlit as st
import gym
from blackjack import BlackjackEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import joblib
from sklearn.naive_bayes import GaussianNB
from PIL import Image

# Função para verificar vencedor
def win(hand, dealer_hand):
    if hand > 21:
        return -1
    elif dealer_hand > 21:
        return 1
    else:
        if hand > dealer_hand:
            return 1
        elif hand < dealer_hand:
            return -1
        else:
            return 0
        
# Carregar o modelo treinado
model = joblib.load('BlackJack/arvore_blackjack.sav')

# Configurar o ambiente
def setup_environment():
    player_2_env = gym.make('Blackjack-v1')
    env = BlackjackEnv(render_mode="human")
    return player_2_env, env

# Função principal para a jogatina
def play_game(player_2_env, env, p1_win, p2_win):

    image_placeholder = st.empty()

    # Requisitos do Player 2 (maquina treinada)
    obs_p2, info = player_2_env.reset()

    # Requisitos do Player 1
    player_2, dealer_value, _ = obs_p2

    # Reset para inicializar as variáveis do ambiente
    obs, info = env.reset(dealer_hand=dealer_value)
    image_array = env.render(player_2)
    image_placeholder.image(Image.fromarray(np.uint8(image_array)))

    done_done = False
    done1 = False

    while not done_done:
        # Jogada do Player 1
        if not done1:
            col1, col2 = st.columns(2)  # Colunas para os botões "HIT" e "STICK"

            with col1:
                hit_button = st.button("HIT")
            with col2:
                stick_button = st.button("STICK")
            while True:
                if hit_button:
                    player_action = 1
                    break
                elif stick_button:
                    player_action = 0
                    break

            obs_p1, reward, terminated, truncated, _ = env.step(player_action)
            image_array = env.render(player_2)
            image_placeholder.image(Image.fromarray(np.uint8(image_array)))
            done1 = terminated or truncated
            
            dealer_hand = env.get_dealer_sum()
        else:
            next_obs, reward, terminated, truncated, _ = player_2_env.step(model.predict([obs_p2])[0])

            image_array = env.render(next_obs[0])
            image_placeholder.image(Image.fromarray(np.uint8(image_array)))

            # Se terminou ou truncou (limite de tempo)
            done_done = terminated or truncated

            obs_p2 = next_obs

    p1_win.append(win(obs_p1[0], dealer_hand))
    p2_win.append(win(obs_p2[0], dealer_hand))

    image_array = env.render(obs_p2[0], done=True)
    image_placeholder.image(Image.fromarray(np.uint8(image_array)))
    # Botão de reiniciar exibido abaixo
    if st.button("Reiniciar"):
        st.experimental_rerun()

# Função para exibir o desempenho
def show_performance(p1_win, p2_win):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(p1_win)), y=np.cumsum(p1_win), label="Player 1 (Humano)")
    sns.lineplot(x=range(len(p2_win)), y=np.cumsum(p2_win), label="Player 2 (Modelo)")
    plt.title("Desempenho Acumulado por Partida")
    plt.xlabel("Número de Partidas")
    plt.ylabel("Vitórias Acumuladas")
    plt.legend()
    st.pyplot(plt)

# Interface Streamlit
st.title("Blackjack - Jogo e Análise de Desempenho")
p1_win = []
p2_win = []

# Menu lateral para escolher entre Jogo ou Desempenho
menu = st.sidebar.radio("Escolha uma opção:", ["Jogo", "Desempenho"])

if menu == "Jogo":
    st.subheader("Jogue Blackjack")
    player_2_env, env = setup_environment()
    play_game(player_2_env, env, p1_win, p2_win)

elif menu == "Desempenho":
    st.subheader("Desempenho dos Jogadores")
    if len(p1_win) > 0 and len(p2_win) > 0:
        show_performance(p1_win, p2_win)
    else:
        st.write("Nenhuma partida jogada ainda. Jogue algumas partidas para visualizar o desempenho.")
