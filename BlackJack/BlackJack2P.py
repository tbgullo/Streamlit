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

def reinicia_game_state():
    player_2_env, env = setup_environment()
    
    # Requisitos do Player 2 (máquina treinada)
    obs_p2, info = player_2_env.reset()
    player_2, dealer_value, _ = obs_p2

    # Reset para inicializar as variáveis do ambiente
    obs_p1, info = env.reset(dealer_hand=dealer_value)

    st.session_state.game_state = {
        "obs_p2": obs_p2,
        "obs_p1": obs_p1,
        "player_2": player_2,
        "dealer_value": dealer_value,
        "dealer_hand": dealer_value,
        "done_done": False,
        "done1": False,
        "player_action": None,
        "player_2_env": player_2_env,
        "env": env,
    }

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

# Variável para contar as chaves
if "key_counter" not in st.session_state:
    st.session_state.key_counter = 10

# Função principal para a jogatina
def play_game(p1_win, p2_win):

    key_counter = st.session_state.key_counter
    st.session_state.key_counter += 1

    if "game_state" not in st.session_state:
        print("reinicio")
        reinicia_game_state()
    
    # Recupera o estado atual
    game_state = st.session_state.game_state
    obs_p2 = game_state["obs_p2"]
    obs_p1 = game_state["obs_p1"]
    player_2 = game_state["player_2"]
    dealer_value = game_state["dealer_value"]
    dealer_hand = game_state["dealer_hand"]
    done_done = game_state["done_done"]
    done1 = game_state["done1"]
    env = game_state["env"]
    player_2_env = game_state["player_2_env"]
    
    image_placeholder = st.empty()
    image_array = env.render(player_2)
    image_placeholder.image(Image.fromarray(np.uint8(image_array)))

    col1, col2 = st.columns(2)  # Colunas para os botões "HIT" e "STICK"

    with col1:
        if st.button("HIT", key=f"hit_button"):
            obs_p1, reward, terminated, truncated, _ = env.step(1)
            game_state["obs_p1"] = obs_p1
            game_state["done1"] = terminated or truncated
            game_state["dealer_hand"] = env.get_dealer_sum()
            image_array = env.render(player_2)
            image_placeholder.image(Image.fromarray(np.uint8(image_array)))

    with col2:
        if st.button("STICK", key=f"stick_button"):
            obs_p1, reward, terminated, truncated, _ = env.step(0)
            game_state["obs_p1"] = obs_p1
            game_state["done1"] = True
            game_state["dealer_hand"] = env.get_dealer_sum()
            image_array = env.render(player_2)
            image_placeholder.image(Image.fromarray(np.uint8(image_array)))

    if done1:
        while not done_done:
                next_obs, reward, terminated, truncated, _ = player_2_env.step(model.predict(np.array(obs_p2).reshape(1, -1))[0])

                image_array = env.render(next_obs[0])
                image_placeholder.image(Image.fromarray(np.uint8(image_array)))

                # Se terminou ou truncou (limite de tempo)
                done_done = terminated or truncated

                obs_p2 = next_obs

        game_state["p1_win"].append(win(obs_p1[0], dealer_hand))
        game_state["p2_win"].append(win(obs_p2[0], dealer_hand))

        image_array = env.render(obs_p2[0], done=True)
        image_placeholder.image(Image.fromarray(np.uint8(image_array)))
        reinicia_game_state()
    
    # Botão de reiniciar exibido abaixo
    if st.button("Reiniciar", key=f"reiniciar_button_{dealer_hand}_{key_counter}"):
        reinicia_game_state()


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
    play_game(p1_win, p2_win)

elif menu == "Desempenho":
    st.subheader("Desempenho dos Jogadores")
    if len(st.session_state.game_state["p1_win"]) > 0 and len(st.session_state.game_state["p2_win"]) > 0:
        show_performance(st.session_state.game_state["p1_win"], st.session_state.game_state["p2_win"])
    else:
        st.write("Nenhuma partida jogada ainda. Jogue algumas partidas para visualizar o desempenho.")
