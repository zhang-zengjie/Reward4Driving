import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import math

def get_unique_numbers(numbers):
    "Get the all action list and note already implemented action taken. Return a list of taken action."

    list_of_unique_numbers = []

    unique_numbers = set(numbers)

    for number in unique_numbers:
        list_of_unique_numbers.append(number)

    return list_of_unique_numbers

    
class Config_PPO:    
    def __init__(self, max_ep_len=200, max_training_timesteps=int(1000), save_model_freq=1e5, K_epochs=50, eps_clip=0.2, lr_actor=1e-4,lr_critic=3e-4,action_std_init=0.6,random_seed=0, gamma=0.995) -> None:
        """ The confiuration of parameters for PPO agent """
        # self.state_size = state_size
        # self.action_size = action_size
        self.has_continuous_action_space = False
        self.max_ep_len = max_ep_len           # max timesteps in one episode
        self.max_training_timesteps = max_training_timesteps
        self.print_freq = max_ep_len * 1
        self.log_freq = max_ep_len * 2         
        self.save_model_freq = save_model_freq    
        self.action_std = 0.6                  
        self.action_std_decay_rate = 0.05      
        self.min_action_std = 0.1              
        self.action_std_decay_freq = int(2.5e5)
        self.update_timestep = max_ep_len * 4      # update policy every n timesteps
        self.K_epochs = K_epochs                   # update policy for K epochs in one PPO update
        self.eps_clip = eps_clip                   # clip parameter for PPO
        self.gamma = gamma                         # discount factor
        self.lr_actor = lr_actor                   # learning rate for actor network
        self.lr_critic = lr_critic                 # learning rate for critic network
        self.random_seed = random_seed             # set random seed if required (0 = no random seed)
        self.action_std_init = action_std_init

class Config_DDPG:
    "The confiuration of parameters for DDPG agent."

    def __init__(self, Episode=5, max_ep_len=700,freq_save_epi=int(500),BUFFER_SIZE= int(1e7), BATCH_SIZE=64,GAMMA=0.95,TAU=1e-3,LR_ACTOR=1e-4,LR_CRITIC=3e-4,WEIGHT_DECAY=0.0001, random_seed=32) -> None:
        
        self.Episode = Episode
        self.max_ep_len = max_ep_len
        self.freq_save_epi = freq_save_epi
        self.BUFFER_SIZE = BUFFER_SIZE # replay buffer size
        self.BATCH_SIZE = BATCH_SIZE   # minibatch size
        self.GAMMA = GAMMA             # discount factor
        self.TAU = TAU                 # for soft update of target parameters
        self.LR_ACTOR = LR_ACTOR       # learning rate of the actor 
        self.LR_CRITIC = LR_CRITIC     # learning rate of the critic
        self.WEIGHT_DECAY = WEIGHT_DECAY # L2 weight decay
        self.random_seed = random_seed

class Config_DQN:

    def __init__(self, save_ep=500,print_ep = 1,n_episodes= 100,SEED=32, BATCHSIZE=64, BUFFERSIZE=int(1e7), UPDATEEVERY=4, GAMMA=0.95, TAU=1e-3, LR= 1e-4,max_t=700, eps_start=1.0, eps_end=0.01, eps_decay=0.996) -> None:
        self.save_ep = save_ep
        self.print_ep = print_ep
        self.n_episodes = n_episodes
        self.seed = SEED
        self.batch_size = BATCHSIZE
        self.buffer_size = BUFFERSIZE
        self.update_every = UPDATEEVERY # 4
        self.gamma = GAMMA # 0.9-0.99
        self.tau = TAU
        self.lr = LR
        self.max_t=max_t
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay = math.exp(math.log(eps_end)/n_episodes)

def score_save(score,averagescore,file_name,path, print_=False):
    """simple save scores to csv as given para"""
    Final_score ={'Score': score,'Averagescores':averagescore}
    df = pd.DataFrame.from_dict(Final_score)
    df.to_csv((path/f"score_{file_name}.csv"))
    if print_:
        return print('file save at :', path, f'file name: score_{file_name}.csv')
    else:
        return