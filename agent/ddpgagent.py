import numpy as np
import random
import copy
from collections import namedtuple, deque
from agent.model_ddpg import Actor2, Critic2
import torch
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent3():
    """Interacts with and learns from the environment."""
    def __init__(self,cfg):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = cfg.state_size
        self.action_size = cfg.action_size
        self.seed = cfg.random_seed
        self.BUFFER_SIZE = cfg.BUFFER_SIZE  # replay buffer size
        self.BATCH_SIZE = cfg.BATCH_SIZE        # minibatch size
        self.GAMMA = cfg.GAMMA            # discount factor
        self.TAU = cfg.TAU              # for soft update of target parameters
        self.LR_ACTOR = cfg.LR_ACTOR         # learning rate of the actor 
        self.LR_CRITIC = cfg.LR_CRITIC        # learning rate of the critic
        self.WEIGHT_DECAY = cfg.WEIGHT_DECAY   # L2 weight decay

    
    # def __init__(self, state_size, action_size, random_seed):
    #     """Initialize an Agent object.
        
    #     Params
    #     ======
    #         state_size (int): dimension of each state
    #         action_size (int): dimension of each action
    #         random_seed (int): random seed
    #     """
    #     self.state_size = state_size
    #     self.action_size = action_size
    #     self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor2(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor2(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic2(self.state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic2(self.state_size, self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            # return self.learn(experiences, self.GAMMA)
            self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(device)
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # input (b, 96,96)
        state =  torch.from_numpy(state).float().flatten().unsqueeze(0).to(device) # flatten before

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()

        # return np.clip(action, -1, 1)
        # return action
        # return np.argmax(np.argmax(action, axis=0))
        return np.argmax(action)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # print(f"state.shape{states.shape}",f"N_state.shape{next_states.shape}" )
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # print(f'AGnext_states.shape{next_states.shape}')
        actions_next = self.actor_target(next_states) #(64,5)
        
        # print(f'AGaction_next.shape{actions_next.shape}\n')
        Q_targets_next = self.critic_target(next_states, actions_next) # (torch.Size([6144, 96]), [64,5])
        # Q_targets_next [64,1]
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # Q_targets [64,1]
        # print(f"AGQ_target\t{Q_targets.shape}")
        # Compute critic loss
        # print(f"AGactions\t{actions.shape} \t states \t{states.shape}")
        actions_ = actions.repeat(1,5)
        Q_expected = self.critic_local(states, actions_) # ([# ([6144,96]), [64,1]])
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2): # as in the papaer
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print("buffer state:",state.shape)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print("buffer state:",state.shape)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)