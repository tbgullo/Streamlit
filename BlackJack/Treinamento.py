import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier as MLP

# Classe de Black Jack

class BlackjackAgent:

    def __init__(self):
        self.X_card = []
        self.y_card = []
        self.model = MLP(max_iter=1500, verbose=False, tol=0.00001, solver='adam', activation='relu', hidden_layer_sizes=(50,50, 25,25 )) #max_iter = epochas, verbose = iteration
        self.learning_Gap = 0
        self.training_error = []        
        self.lose = 0
        self.win = 0

    def get_action(self, observation):

        if (self.learning_Gap <= 80):

            if (observation[0] < 17):
                return 1
            
            return 0
        
        return self.model.predict(np.array(observation).reshape(1, -1))[0]

    def update(self, observation, reward, done, action):
            
            # A melhor decisão

            if reward == -1:

                # Se o modelo pediu para parar e perdeu por soma do dealer > player então deveria ter continuado
                if action == 0:
                    self.y_card.append(1)

                # Se o modelo pediu para continuar mas burstou então deveria ter parado
                else:
                    self.y_card.append(0)


            # Se teve empate (ainda não acabou) ou venceu então o modelo acertou
            if reward >= 0:
                self.y_card.append(action)


            self.X_card.append(observation)
            
            if (self.learning_Gap % 80 == 0 and self.learning_Gap > 0):

                self.model.fit(self.X_card, self.y_card)

            if(done):
                self.learning_Gap += 1

                if (reward == -1):
                    self.lose +=1
                if (reward > 0):
                    self.win +=1

            
            self.training_error.append(self.lose)
        
        
# Preparano o Ambiente 
env = gym.make('Blackjack-v1', render_mode = "human", natural=True, sab=False)
env.reset()
env.render()

# Loop de Treinamento

agent = BlackjackAgent()
n_episodes = 2000
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    
    # para cada partida
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Se terminou ou truncou (limite de tempo)
        done = terminated or truncated

        # Atualiza o agente
        agent.update(obs, reward, done, action)

        obs = next_obs

env.close()    
print("Treinamento Concluido")

# mostra Resumo do Treinamento

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

axs[0].set_title("Penalidade por tentativas")
axs[0].plot(range(len(agent.training_error)), agent.training_error)
axs[0].set_ylabel("Penalidade")
axs[0].set_xlabel("Tentativa")

# Criar os dados para o gráfico
data = {'Result': ['Win', 'Lose', 'Draw'], 'Count': [agent.win, agent.lose, n_episodes - (agent.win + agent.lose)]}

# Plotar o gráfico de barras usando seaborn
axs[1] = sns.barplot(x='Result', y='Count', data=data)

# Adicionar rótulos nas barras
for index, value in enumerate(data['Count']):
    axs[1].text(index, value + 1, str(value), ha='center', fontweight='bold')
    
# Adicionando legendas e rótulos
axs[1].set_xlabel("Resultado")
axs[1].set_ylabel("Número de Partidas")
axs[1].set_title("Número de Vitórias vs Derrotas")

plt.tight_layout()
plt.show()

# Montar arvore de decisão
from sklearn.tree import DecisionTreeClassifier

classificador_BJ = DecisionTreeClassifier(criterion ='entropy')
classificador_BJ.fit(agent.X_card, agent.y_card)

import pickle

#pickle.dump(classificador_BJ, open('../Streamlit/arvore_blackjack.sav', 'wb'))
pickle.dump(agent.model, open('mlp_blackjack.sav', 'wb'))