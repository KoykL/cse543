import os.path
import pickle
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
    def __init__(self, model, optimizer_path):
        self.model = model
        self.optimizer_path = optimizer_path
        self.model.train(True)
        self.optimizer = Adam(self.model.parameters())
        self.load_optimizer()

    def save_optimizer(self):
        print("save optimizer")
        torch.save(self.optimizer.state_dict(), self.optimizer_path)

    def load_optimizer(self):
        if os.path.exists(self.optimizer_path):
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            print("load optimizer info")
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
        return loss

class DQLTrainer(object):
    def __init__(self, model_path, optimizer_path, memory_path):
        self.memory_path = memory_path
        self.memory = ReplayMemory(1000)
        self.load_memory()

        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.optimizer = DQLOptimizer(get_model(self.model_path), self.optimizer_path)

    def save_memory(self):
        with open(self.memory_path, "wb") as f:
            print("save memory")
            pickle.dump(self.memory, f)

    def load_memory(self):
        if os.path.exists(self.memory_path):
            print("load memory")
            with open(self.memory_path, "rb") as f:
                self.memory = pickle.load(f)
    def run_iter(self):
        print("running one iteration")
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
            print("agent {} played: {}".format(agent_playing, action))
            for i, a_s in enumerate(platform.game_state.agent_states):
                print("agent {} has card: {}".format(i, a_s.get_cards_str()))
        history.winner = platform.game_state.who_wins()
        history.append_memories(self.memory)
        for agent in agents:
            agent.terminate()

        for i in range(100):
            print("optimization iteration", i)
            loss = self.optimizer.run_iter(self.memory)
            print("loss", loss)

        save_model(self.optimizer.model, self.model_path)
        self.optimizer.save_optimizer()
        self.save_memory()
