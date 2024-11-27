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
import pandas as pd

log_placeholder = st.empty()
# Função para atualizar o log na interface
def log_message(message):
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(message)
    log_placeholder.text("\n".join(st.session_state.logs))

def reinicia_game_state(p1_win = [], p2_win = []):
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
        "p1_win" : p1_win,
        "p2_win" : p2_win,
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
def play_game():

        key_counter = st.session_state.key_counter
        st.session_state.key_counter += 1

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

                
        obs_p2 = game_state["obs_p2"]
        obs_p1 = game_state["obs_p1"]
        player_2 = game_state["player_2"]
        dealer_value = game_state["dealer_value"]
        dealer_hand = game_state["dealer_hand"]
        done_done = game_state["done_done"]
        done1 = game_state["done1"]
        env = game_state["env"]
        player_2_env = game_state["player_2_env"]

        if done1:

            while not done_done:
                    #next_obs, reward, terminated, truncated = None, None, None, None

                    action = model.predict(np.array(obs_p2).reshape(1, -1))[0]
                    try:
                        next_obs, reward, terminated, truncated, _ = player_2_env.step(action)
                    except AttributeError as e:
                        x = 0
                    next_obs, reward, terminated, truncated, _ = player_2_env.step(action)
                    image_array = env.render(next_obs[0])

                    image_placeholder.image(Image.fromarray(np.uint8(image_array)))

                    # Se terminou ou truncou (limite de tempo)
                    done_done = terminated or truncated

                    obs_p2 = next_obs

            game_state["p1_win"].append(win(obs_p1[0], dealer_hand))
            game_state["p2_win"].append(win(obs_p2[0], dealer_hand))

            image_array = env.render(obs_p2[0], done=True)
            image_placeholder.image(Image.fromarray(np.uint8(image_array)))
            reinicia_game_state(game_state["p1_win"],game_state["p2_win"])
            
            if win(obs_p1[0], dealer_hand) == 1:
                status_placeholder.markdown(f"<h2 style='text-align: center;'>Voce Ganhou</h2>", unsafe_allow_html=True)
            if win(obs_p1[0], dealer_hand) == -1:
                status_placeholder.markdown(f"<h2 style='text-align: center;'>Voce Perdeu</h2>", unsafe_allow_html=True)
            if win(obs_p1[0], dealer_hand) == 0:
                status_placeholder.markdown(f"<h2 style='text-align: center;'>Empatou</h2>", unsafe_allow_html=True)
        
        # Botão de reiniciar exibido abaixo
        if st.button("Reiniciar", key=f"reiniciar_button_{dealer_hand}_{key_counter}"):
            reinicia_game_state(game_state["p1_win"],game_state["p2_win"])



# Função para exibir o desempenho
def show_performance(p1_win, p2_win):

    st.subheader("Desempenho dos Jogadores")

    st.subheader("Desempenho Historico")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(p1_win)), y=p1_win, label="Player 1 (Humano)")
    sns.lineplot(x=range(len(p2_win)), y=p2_win, label="Player 2 (Modelo)")
    plt.title("Desempenho por Partida")
    plt.xlabel("Partida")
    plt.ylabel("Vitórias/Derrota")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Desempenho Geral")

    # Contagem de vitórias, derrotas e empates para o jogador 1 (p1_win)
    p1_victories = sum([1 for x in p1_win if x > 0])  # Contando as vitórias
    p1_losses = sum([1 for x in p1_win if x < 0])  # Contando as derrotas
    p1_draws = sum([1 for x in p1_win if x == 0])  # Contando os empates

    # Contagem de vitórias, derrotas e empates para o jogador 2 (p2_win)
    p2_victories = sum([1 for x in p2_win if x > 0])  # Contando as vitórias
    p2_losses = sum([1 for x in p2_win if x < 0])  # Contando as derrotas
    p2_draws = sum([1 for x in p2_win if x == 0])  # Contando os empates

    data = {
        "Resultado": ["Vitórias", "Derrotas", "Empates"],
        "Você (Player 1)": [p1_victories, p1_losses, p1_draws],
        "Player 2": [p2_victories, p2_losses, p2_draws]
    }

    # Criando o gráfico
    df = pd.DataFrame(data)
    df.set_index("Resultado", inplace=True)

    # Plotando gráfico de barras
    df.plot(kind="bar", figsize=(10, 6), color=['blue', 'red'])
    plt.title("Vitórias, Derrotas e Empates por Jogador")
    plt.xlabel("Resultado")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    st.pyplot(plt)
# Interface Streamlit
st.title("Blackjack - Jogo e Análise de Desempenho")

if "game_state" not in st.session_state:
        log_message("reinicio")
        reinicia_game_state([],[])

# Menu lateral para escolher entre Jogo ou Desempenho
menu = st.sidebar.radio("Escolha uma opção:", ["Jogo", "Desempenho"])

if menu == "Jogo":
    st.subheader("Jogue Blackjack")
    # Exibir o status do jogador embaixo, centralizado
    status_placeholder = st.empty()  # Criar um espaço vazio para o status
    status_placeholder.markdown(f"<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)
    play_game()

elif menu == "Desempenho":

    if len(st.session_state.game_state["p1_win"]) > 0 and len(st.session_state.game_state["p2_win"]) > 0:
        show_performance(st.session_state.game_state["p1_win"], st.session_state.game_state["p2_win"])
    else:
        st.write("Nenhuma partida jogada ainda. Jogue algumas partidas para visualizar o desempenho.")
