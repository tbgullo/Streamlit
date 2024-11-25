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
    # Inicializa o estado do jogo
    if "game_state" not in st.session_state:
        # Requisitos do Player 2 (máquina treinada)
        obs_p2, info = player_2_env.reset()
        player_2, dealer_value, _ = obs_p2

        # Reset para inicializar as variáveis do ambiente
        obs, info = env.reset(dealer_hand=dealer_value)
        st.session_state.game_state = {
            "obs_p2": obs_p2,
            "obs": obs,
            "player_2": player_2,
            "dealer_value": dealer_value,
            "done_done": False,
            "done1": False,
            "p1_win": p1_win,
            "p2_win": p2_win,
        }

    # Recupera o estado atual
    game_state = st.session_state.game_state
    obs_p2 = game_state["obs_p2"]
    obs = game_state["obs"]
    player_2 = game_state["player_2"]
    dealer_value = game_state["dealer_value"]
    done_done = game_state["done_done"]
    done1 = game_state["done1"]

    # Placeholder para a imagem do jogo
    image_placeholder = st.empty()
    image_array = env.render(player_2)
    image_placeholder.image(Image.fromarray(np.uint8(image_array)))

    # Loop do jogo
    if not done_done:
        if not done1:
            col1, col2 = st.columns(2)  # Botões "HIT" e "STICK" em linha

            with col1:
                hit_button = st.button("HIT", key="hit")
            with col2:
                stick_button = st.button("STICK", key="stick")

            if hit_button:
                player_action = 1
                obs, reward, terminated, truncated, _ = env.step(player_action)
                game_state["obs"] = obs
                game_state["done1"] = terminated or truncated
                game_state["dealer_value"] = env.get_dealer_sum()
            elif stick_button:
                player_action = 0
                obs, reward, terminated, truncated, _ = env.step(player_action)
                game_state["obs"] = obs
                game_state["done1"] = terminated or truncated
                game_state["dealer_value"] = env.get_dealer_sum()

        else:
            next_obs, reward, terminated, truncated, _ = player_2_env.step(model.predict([obs_p2])[0])
            game_state["obs_p2"] = next_obs
            game_state["done_done"] = terminated or truncated

    else:
        game_state["p1_win"].append(win(obs[0], dealer_value))
        game_state["p2_win"].append(win(obs_p2[0], dealer_value))
        image_array = env.render(obs_p2[0], done=True)
        image_placeholder.image(Image.fromarray(np.uint8(image_array)))

    # Botão de reiniciar
    if st.button("Reiniciar"):
        del st.session_state["game_state"]
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
    if st.button("Iniciar Partida"):
        play_game(player_2_env, env, p1_win, p2_win)

elif menu == "Desempenho":
    st.subheader("Desempenho dos Jogadores")
    if len(p1_win) > 0 and len(p2_win) > 0:
        show_performance(p1_win, p2_win)
    else:
        st.write("Nenhuma partida jogada ainda. Jogue algumas partidas para visualizar o desempenho.")
