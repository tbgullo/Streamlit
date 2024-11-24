import gym
from blackjack import BlackjackEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB

# função para ver se ganhou
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

# Configurando Ambientes
player_2_env = gym.make('Blackjack-v1', render_mode = "human", natural=True, sab=False)
env = BlackjackEnv(render_mode="human")

# Modelo Final de arquivo pkl
import pickle
model = pickle.load(open('../BlackJack/arvore_blackjack.sav', 'rb'))

#teste 
import keyboard

# Guardar as informações
p1_win = []
p2_win = []
dealer_hand = 0

## Parte da Jogatina

play = True

while play is True:

    # Requisitos do Player 2 (maquina treinada)
    obs_p2, info = player_2_env.reset()

    # Requisitos do Player 1
    player_2, dealer_value, _ = obs_p2

    # Reset para inicializar as variáveis do ambiente
    obs, info = env.reset(dealer_hand=dealer_value)
    env.render(player_2)
    
    done_done = False
    done1 = False

    # para cada partida
    while not done_done:

        #Primeiro Jogatina do Player 1 com tecla
        if not done1:
            #imaginando que se clicar S o valor eh 0 e se clicar H o valor é 1
            print("Sua vez! Pressione 'H' para HIT ou 'S' para STAND.")

            # Aguarda a ação do jogador
            while True:
                if keyboard.is_pressed('H'):
                    player_action = 1  # HIT
                    break
                elif keyboard.is_pressed('S'):
                    player_action = 0  # STAND
                    break
            #player_action = algum deles 

            obs_p1, reward, terminated, truncated, _ = env.step(player_action)
            env.render(player_2)
            done1 = terminated or truncated

            dealer_hand = env.get_dealer_sum()

        else:
            next_obs, reward, terminated, truncated, _ = player_2_env.step(model.predict([obs_p2])[0])
            env.render(next_obs[0])
            # Se terminou ou truncou (limite de tempo)
            done_done = terminated or truncated

            obs_p2 = next_obs
    
    env.render(obs_p2[0], done=True)

    p1_win.append(win(obs_p1[0], dealer_hand))
    p2_win.append(win(obs_p2[0], dealer_hand))

    # Botao de reiniciar na tecla F
    # quando clicado volta, se não fica em loop infinito esperando reiniciar
    print("Partida finalizada. Pressione 'F' para reiniciar ou 'Q' para sair.")
    
    # Aguarda o comando para reiniciar ou sair
    while True:
        if keyboard.is_pressed('F'):
            break  # Sai do loop interno e reinicia a partida
        elif keyboard.is_pressed('Q'):
            play = False  # Encerra o jogo
            break
