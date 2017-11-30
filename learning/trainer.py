import random
from collections import namedtuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam

from core.agent import DQLAgent
from core.platform import PrivateGameState, Platform
from learning.network import get_model, save_model

Transition = namedtuple('Transition',
                        ('state', 'priors', 'reward'))
BATCH_SIZE = 32
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class GameHistoryFactory(object):
    def __init__(self):
        self.priors = []
        self.states = []
        self.winner = None

    def append_step(self, state, priors):
        self.priors.append(priors)
        self.states.append(state)

    def append_result(self, winner):
        self.winner = winner

    def append_memories(self, memory):
        for state, prior in zip(self.priors, self.states):
            other_player = -1
            for i in range(3):
                if i != state.whos_turn and i != 0:
                    other_player = i
                    break
            if state.whos_turn == self.winner or (
                    (not state.agent_state.isLandlord) and state.whos_turn == other_player):
                reward = 1
            else:
                reward = -1
            memory.push(state, prior, reward)


class DQLOptimizer(object):
    def __init__(self, model):
        self.model = model
        self.model.train(True)
        self.optimizer = Adam(self.model.parameters())

    def run_iter(self, memory):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        priors_batch = Variable(torch.cat(batch.priors))
        reward_batch = Variable(torch.cat(batch.reward))

        priors, value = self.model(state_batch)
        # Compute Huber loss

        loss = (value - reward_batch) ^ 2 - priors_batch * torch.log(priors)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQLTrainer(object):
    def __init__(self, model_path):
        self.memory = ReplayMemory(1000)
        self.model_path = model_path

    def run_iter(self):
        agents = [DQLAgent(i, self.model_path, True) for i in range(3)]
        for agent in agents:
            agent.start()
        platform = Platform(agents)

        platform.deal()
        history = GameHistoryFactory()
        while not platform.game_state.isTerminal():
            agent_playing = platform.game_state.whos_turn
            state = platform.game_state.getPrivateStateForAgentX(platform.game_state.whos_turn).agent_state
            agent = platform.agents[platform.game_state.whos_turn]
            if agent.t is not None:
                reverse_map = PrivateGameState.getAllActionsReverseMap(len(agent.t.root.state.state.agent_state.cards))
                all_action_nums = PrivateGameState.max_combinations(len(agent.t.root.state.state.agent_state.cards))
                all_actions = np.zeros(all_action_nums)
                for a, c in zip(agent.t.root.actions, agent.t.root.children):
                    if not a.is_pass:
                        action_tuple = tuple(sorted(a.idx))
                        all_actions[reverse_map[action_tuple]] = c.play_count
                    else:
                        all_actions[-1] = c.play_count

                all_actions = np.array(all_actions)
                all_actions /= all_actions.sum()
                history.append_step(state, all_actions)
            action = platform.turn()

        history.winner = platform.game_state.who_wins()
        history.append_memories(self.memory)
        for agent in agents:
            agent.terminate()

        model = get_model(self.model_path)
        optimizer = DQLOptimizer(model)
        for i in range(100):
            optimizer.run_iter(self.memory)
        save_model(model, self.model_path)
