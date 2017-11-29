import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state

    for t in count():
        # Select and perform an action
        action = select_action(state)

        _, reward, done, _ = env.step(action[0, 0])

        reward = Tensor([reward])

        # Observe new state

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


class DeepLearner(object):
    def __init__(self, model):
        self.model = model

    def _gen_last_dealt_hand(self, private_state):
        dealt_hand = np.zeros(1, 54)
        for card in private_state.last_dealt_hand:
            seq = card.seq()
            if seq < 13:
                color_seq = card.color_seq()
                dealt_hand[0, seq * 13 + color_seq] = 1
            else:
                dealt_hand[0, 52 + seq - 13] = 1
        return dealt_hand

    def _gen_hand(self, private_state):
        hand = np.zeros(1, 54)
        for card in private_state.agent_state.cards:
            seq = card.seq()
            if seq < 13:
                color_seq = card.color_seq()
                hand[0, seq * 13 + color_seq] = 1
            else:
                hand[0, 52 + seq - 13] = 1
        return hand

    def _gen_other_cards(self, private_state):
        cards = np.zeros(1, 54)
        for card in private_state.agent_state.other_cards:
            seq = card.seq()
            if seq < 13:
                color_seq = card.color_seq()
                cards[0, seq * 13 + color_seq] = 1
            else:
                cards[0, 52 + seq - 13] = 1
        return cards

    def _gen_landlord(self, private_state):
        landlord = private_state.agent_state.cards.isLandlord
        land_tensor = np.full((1, 54), landlord)
        return land_tensor

    def _gen_landlord_guard(self, private_state):
        guard = private_state.x == 1
        return np.full((1, 54), guard)

    def _gen_is_last_pass(self, private_state):
        last_pass = private_state.pass_count == 1
        return np.full((1, 54), last_pass)

    def _gen_other_agents_number_cards(self, private_state):
        cards = np.zeros((3, 54))
        for i, card_num in enumerate(private_state.agent_num_cards):
            cards[i] = card_num
        return cards

    def _gen_input(self, private_state):
        last_dealt_hand = self._gen_last_dealt_hand(private_state)
        last_hand = self._gen_hand(private_state)
        other_hand = self._gen_other_cards(private_state)
        landload = self._gen_landlord(private_state)
        landload_guard = self._gen_landlord_guard(private_state)
        last_pass = self._gen_is_last_pass(private_state)
        other_cards = self._gen_other_agents_number_cards(private_state)
        return np.concatenate(
            (last_dealt_hand, last_hand, other_hand, landload, landload_guard, last_pass, other_cards), axis=0)

    # def get_action(self, net_out):
    #     indices = np.array(PrivateGameState.getAllActions())
    #     vals, sel_indices = torch.max(net_out, 0)
    #     return indices[sel_indices[0]]
    #
    # def estimate_actions(self, private_state):
    #     net_in = self._gen_input(private_state)
    #     net_out = self.model(net_in)
    #     card_indices = self.get_action(net_out)
    #     return card_indices
    def estimate_leaf_prior_value(self, private_state):
        net_in = self._gen_input(private_state)
        prior, value = self.model(net_in)
        return prior, value
