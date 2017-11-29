import random
from collections import namedtuple

from core.platform import PrivateGameState
from learning.vdcnn.VDCNN import VDCNN

net = VDCNN(input_dim=9)
Transition = namedtuple('Transition',
                        ('state', 'priors', 'reward'))


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


def play(platform, verbose=False):
    # start game
    platform.deal()
    history = GameHistoryFactory()
    memory = ReplayMemory(10000)
    while not platform.game_state.isTerminal():
        agent_playing = platform.game_state.whos_turn
        state = platform.agent_states[platform.game_state.whos_turn].getPrivateStateForAgentX(
            platform.game_state.whos_turn)
        agent = platform.agents[platform.game_state.whos_turn]
        # priors = [c.play_count for c in agent.t.children]
        reverse_map = PrivateGameState.getAllActionsReverseMap(len(agent.t.root.state.state.agent_state.cards))
        all_actions = PrivateGameState.getAllActions(len(agent.t.root.state.state.agent_state.cards))
        for a, c in zip(agent.t.root.actions, agent.t.root.children):
            action_tuple = tuple(sorted(a.idx))
            all_actions[reverse_map[action_tuple]] = c.play_count
        history.append_step(state, all_actions)
        action = platform.turn()
    history.winner = platform.game_state.who_wins()
    history.append_memories(memory)
