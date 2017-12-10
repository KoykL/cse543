import os.path
import pickle
import queue
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional
from torch.autograd import Variable
from torch.optim import Adam

from core.agent import DQLAgent
from core.platform import PrivateGameState, Platform
from learning.network import get_model, save_model, DeepLearner

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
        for prior, state in zip(self.priors, self.states):
            other_player = -1
            for i in range(3):
                if i != state.whos_turn and i != 0:
                    other_player = i
                    break
            if state.whos_turn == self.winner or (
                    (not state.x == 0) and self.winner == other_player):
                reward = 1
            else:
                reward = -1
            memory.push(state, prior, reward)


class DQLOptimizer(object):
    def __init__(self, model, optimizer_path):
        self.model = model
        self.optimizer_path = optimizer_path
        self.model.train(True)
        self.optimizer = Adam(self.model.parameters(), weight_decay=1e-4)
#        self.optimizer = SGD(self.model.parameters(), lr = 0.0001, momentum=0.9, weight_decay=1e-5)
        self.load_optimizer()
#        self.optimizer = LBFGS(self.model.parameters())
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
        learner = DeepLearner(self.model)
        states = []
        for state in batch.state:
            states.append(learner._gen_input(state))
        states_raw = FloatTensor(states)
        priors_raw = FloatTensor(list(batch.priors))
        rewards_raw = FloatTensor(list(batch.reward))
        state_batch = Variable(states_raw)
        priors_batch = Variable(priors_raw)
        reward_batch = Variable(rewards_raw)
        def closure():
            self.optimizer.zero_grad()
            priors, value = self.model(state_batch)
            priors = torch.nn.functional.log_softmax(priors)
            loss = torch.sum(torch.pow((torch.squeeze(value) - reward_batch), 2)) - torch.sum(priors_batch * priors)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            return loss
        self.optimizer.step(closure)
        priors, value = self.model(state_batch)
        priors = torch.nn.functional.log_softmax(priors)
        loss_val = torch.sum(torch.pow((torch.squeeze(value) - reward_batch), 2))
        loss_dist = - torch.sum(priors_batch * priors)
        loss = loss_val + loss_dist
        return loss, loss_val, loss_dist

class DQLTrainer(object):
    def __init__(self, model_path, optimizer_path, memory_path):
        self.memory_path = memory_path
        self.memory = ReplayMemory(10000)
        self.load_memory()

        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.optimizer = DQLOptimizer(get_model(self.model_path), self.optimizer_path)
        #self.optimizer.load_optimizer()
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
        # for _ in range(5):
        #     print("running one game")
        #     turns = 5
        #     agents = [DQLAgent(i, self.model_path, True, turns=turns) for i in range(3)]
        #     for agent in agents:
        #         agent.start()
        #         platform = Platform(agents)

        #     platform.deal()
        #     history = GameHistoryFactory()
        #     while not platform.game_state.isTerminal():
        #         agent_playing = platform.game_state.whos_turn
        #         state = platform.game_state.getPrivateStateForAgentX(platform.game_state.whos_turn)
        #         agent = platform.agents[platform.game_state.whos_turn]
        #         action = platform.agents[platform.game_state.whos_turn].getAction(state)
        #         counter = 0
        #         while True:
        #             newroot = agent.states.get()
        #             #print("back", " ".join(str(c) for c in root.state.state.agent_state.cards))
        #             print(counter)
        #             if newroot.state.state == state:
        #                 root = newroot
        #                 counter += 1
        #                 if counter >= turns:
        #                     break                
        #                 #print(" ".join(str(c) for c in root.state.state.agent_state.cards))
        #                 #print(" ".join(str(c) for c in state.agent_state.cards))
            
        #         if root.state.state == state:
        #             reverse_map = PrivateGameState.getAllActionsReverseMap(len(root.state.state.agent_state.cards))
        #             all_action_nums = sum(PrivateGameState.max_combinations())
        #             print(all_action_nums)
        #             all_actions = np.zeros(all_action_nums + 1)
        #             for a, c in zip(root.actions, root.children):
        #                 if not a.is_pass:
        #                     action_tuple = tuple(sorted(a.idx))
        #                     all_actions[reverse_map[action_tuple]] = c.play_count
        #                 else:
        #                     all_actions[-1] = c.play_count
        #             all_actions = np.array(all_actions)
        #             all_actions /= all_actions.sum()
        #             history.append_step(state, all_actions)
        #             print("apend, state", state)
        #         platform.game_state = platform.game_state.getNewState(action)
        #         for agent in platform.agents:
        #             agent.postAction(action)
        #         print("agent {} played: {}".format(agent_playing, action))
        #         for i, a_s in enumerate(platform.game_state.agent_states):
        #             print("agent {} has card: {}".format(i, a_s.get_cards_str()))
        #     history.winner = platform.game_state.who_wins()
        #     history.append_memories(self.memory)
        #     print(len(self.memory))
        #     for agent in agents:
        #         agent.terminate()
        #     self.save_memory()

        for i in range(1000):
            print("optimization iteration", i)
            loss, loss_val, loss_dist = self.optimizer.run_iter(self.memory)
            print("loss", loss, loss_val, loss_dist)

        save_model(self.optimizer.model, self.model_path)
        self.optimizer.save_optimizer()

