import math
import random
import functools
from copy import deepcopy

from core.platform import Platform, AgentState, PrivateGameState


class Node(object):
    def __init__(self, info_set, parent):
        self.actions = []
        self.children = []
        self.parent = parent
        self.state = info_set

        self.empirical_reward = 0
        self.play_count = 1

        self.availability_count = 1

class InformationSet:
    def __init__(self, private_state):
        self.state = private_state
    def __eq__(self, other):
        # print('info comp')
        return self.state == other.state
    def determinization(self):
        deck = list(self.state.other_cards)
        random.shuffle(deck)
        agent1 = AgentState(deck[:self.state.agent_num_cards[(self.state.x + 1) % 3]])
        agent3 = AgentState(deck[:self.state.agent_num_cards[(self.state.x + 2) % 3]])
        if self.state.x == 1:
            agent1, agent3 = agent3, agent1
        instantiations = [agent1, agent3]
        return self.state.getPublicInstantiation(instantiations)

    def is_terminal(self):  # fix it
        return self.state.isTerminal()

    def getNewState(self, act):
        return InformationSet(self.state.getNewState(act, self.state.whos_turn))

    def is_compatible(self, state):
        raise NotImplementedError()
        pass

class Tree(object):
    def __init__(self, private_state=None):
        self.root = Node(InformationSet(private_state), None)


    @staticmethod
    def ucb_val(node):
        k = 1
        return (node.empirical_reward / node.play_count) + k * math.sqrt(
            math.log(node.availability_count) / node.play_count)
    def run_iter(self):
        curr_node = self.root
        curr_determined_state = curr_node.state.determinization()

        candidate_actions = []
        candidate_states = []
        candidate_determined_states = []

        cached_available_nodes = []
        def check_available_children(c, states):
            for state in states:
                if c.state == state:
                    return True
            return False

        def check_in_children(state, curr_node):
            for child in curr_node.children:
                if child.state == state:
                    return True
            return False
        while True:
            # print("descend")
            if curr_node.state.is_terminal():
                break
            actions = curr_determined_state.getPrivateStateForAgentX(curr_determined_state.whos_turn).getLegalActions()
            determined_states = [curr_determined_state.getNewState(act) for act in actions]
            states = [curr_node.state.getNewState(act) for act in actions]
            available_children_mask = list(map(lambda x: check_available_children(x, states), curr_node.children))
            cached_available_nodes.append(available_children_mask)
            # info_states = [InformationSet(s) for s in states]
            states_mask = list(map(lambda x: not check_in_children(x, curr_node), states))
            new_states = list(map(lambda c: c[1], filter( lambda c: states_mask[c[0]],enumerate(states))))
            new_actions = list(map(lambda c: c[1], filter( lambda c: states_mask[c[0]],enumerate(actions))))
            if len(new_states) > 0:
                candidate_actions = new_actions
                candidate_states = new_states
                candidate_determined_states = determined_states
                break
            else:
                # print("children len", len(curr_node.children), len(actions))
                children_vals = [(i, Tree.ucb_val(c)) for i, c in
                                 filter(lambda c: available_children_mask[c[0]], enumerate(curr_node.children))]
                chosen_child = min(children_vals, key=lambda x: x[1])
                chosen_child = chosen_child[0]
                curr_determined_state = curr_determined_state.getNewState(curr_node.actions[chosen_child])
                curr_node = curr_node.children[chosen_child]


        # expansion
        if not curr_node.state.is_terminal():
            chosen_child = random.randint(0, len(candidate_actions) - 1)
            # print("fix this")
            new_s = candidate_states[chosen_child]
            new_c = Node(new_s, curr_node)
            curr_node.actions.append(candidate_actions[chosen_child])
            curr_node.children.append(new_c)
            states = [curr_node.state.getNewState(act) for act in actions]
            available_children_mask = list(map(lambda x: check_available_children(x, states), curr_node.children))
            cached_available_nodes[-1] = available_children_mask

            curr_node = new_c
            curr_determined_state = candidate_determined_states[chosen_child]


        # simulation only curr_determined_state is updated
        while not curr_determined_state.isTerminal():
            actions = curr_determined_state.getPrivateStateForAgentX(curr_determined_state.whos_turn).getLegalActions()
            chosen_child = random.randint(0, len(actions) - 1)
            curr_determined_state = curr_determined_state.getNewState(actions[chosen_child])
        winner = curr_determined_state.who_wins()
        # run until termination

        # backpropagation
        curr_available_children_idx = -1
        # print(cached_available_nodes)
        while curr_node != None:
            #
            if (self.root.state.state.agent_state.isLandlord and winner == self.root.state.state.whos_turn) or (not self.root.state.state.agent_state.isLandlord and (winner == 1 or winner == 2)):
                curr_node.empirical_reward += 1
            curr_node.play_count += 1
            curr_node = curr_node.parent
            if curr_node != None:
                # print("cached", cached_available_nodes)
                for i, child in enumerate(curr_node.children):
                    # print("curr", curr_available_children_idx)
                    if cached_available_nodes[curr_available_children_idx][i]:
                        child.availability_count += 1
            curr_available_children_idx -= 1