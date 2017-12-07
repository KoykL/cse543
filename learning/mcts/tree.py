import random
from copy import deepcopy
import math
import numpy as np

from core.platform import AgentState, PrivateGameState


class Node(object):
    def __init__(self, info_set, parent):
        self.actions = []
        self.children = []
        self.parent = parent
        self.state = info_set

        self.empirical_reward = 0
        self.play_count = 1

        self.availability_count = 0

        self.children_prior = []
        # self.prior = 0

class InformationSet:
    def __init__(self, private_state):
        self.state = private_state

    def __eq__(self, other):
        # print('info comp')
        return self.state == other.state

    def determinization(self):
        deck = list(self.state.other_cards)
        random.shuffle(deck)
        assert self.state.agent_num_cards[(self.state.x + 1) % 3] + self.state.agent_num_cards[
            (self.state.x + 2) % 3] == len(deck)
        agent1 = AgentState(deck[:self.state.agent_num_cards[(self.state.x + 1) % 3]])
        agent3 = AgentState(deck[self.state.agent_num_cards[(self.state.x + 1) % 3]:self.state.agent_num_cards[
                                                                                        (self.state.x + 1) % 3] +
                                                                                    self.state.agent_num_cards[
                                                                                        (self.state.x + 2) % 3]])
        # print("infoset")
        # print(self.state.agent_num_cards[(self.state.x + 1) % 3])
        # print(self.state.agent_num_cards[(self.state.x + 2) % 3])
        # print(agent1.get_cards_str())
        # print(agent3.get_cards_str())
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
    def __init__(self, learner, private_state=None):
        self.root = Node(InformationSet(private_state), None)
        self.learner = learner
        self.iterations_per_determinization = 0
        self.determinization = None

    @staticmethod
    def net_val(node, noise=None):
        return Tree.net_val_without_node(node.empirical_reward, node.play_count, node.availability_count, node.prior, noise)

    @staticmethod
    def net_val_without_node(empirical_reward, play_count, availability_count, prior, noise=None):
        #noise = np.random.dirichlet((0.03,))
        epsilon = 0.25
        if noise is not None:
            return empirical_reward / play_count + ((1 - 0.25) * prior + 0.25*noise) * math.sqrt(availability_count) / play_count
        else:
            return empirical_reward / play_count + prior * math.sqrt(availability_count) / play_count
        
    def run_iter(self):
        curr_node = self.root
#        print("run iter")
        if self.determinization is None or self.iterations_per_determinization >= 100:
            curr_determined_state = curr_node.state.determinization()
            self.determinization = deepcopy(curr_determined_state)
            self.iterations_per_determinization = 0
            # print("change determinization")
        else:
            curr_determined_state = deepcopy(self.determinization)
            self.iterations_per_determinization += 1

        candidate_actions = []
        candidate_states = []

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
        cached_available_nodes = []
        while True:
            if curr_node.state.is_terminal():
                break
            actions = curr_determined_state.getPrivateStateForAgentX(curr_determined_state.whos_turn).getLegalActions()
            # print("hand:", curr_determined_state.getPrivateStateForAgentX(curr_determined_state.whos_turn).agent_state.get_cards_str())
            # print("agent num:", curr_determined_state.whos_turn)
            states = [curr_node.state.getNewState(act) for act in actions]
            available_children_mask = list(map(lambda x: check_available_children(x, states), curr_node.children))
            cached_available_nodes.append(available_children_mask)
            # info_states = [InformationSet(s) for s in states]
            states_mask = list(map(lambda x: not check_in_children(x, curr_node), states))
            new_states = list(map(lambda c: c[1], filter(lambda c: states_mask[c[0]], enumerate(states))))
            new_actions = list(map(lambda c: c[1], filter(lambda c: states_mask[c[0]], enumerate(actions))))

            total_noises = np.random.dirichlet(tuple([0.03]*(len(curr_node.children)+len(new_actions)) ))
            children_vals = [(i, Tree.net_val(c, total_noises[i])) for i, c in filter(lambda c: available_children_mask[c[0]], enumerate(curr_node.children))]
            if len(children_vals) > 0:
                chosen_child = max(children_vals, key=lambda x: x[1])
                chosen_child_val = chosen_child[1]
                chosen_child = chosen_child[0]
            else:
                chosen_child_val = float('-inf')
            if len(new_states) == 0:
                curr_determined_state = curr_determined_state.getNewState(curr_node.actions[chosen_child])
                curr_node = curr_node.children[chosen_child]
            else:
                curr_player_states = curr_determined_state.getPrivateStateForAgentX(curr_determined_state.whos_turn)

                mask_length = sum(PrivateGameState.max_combinations())+1
                mask = np.zeros(mask_length, dtype=np.bool)
                reverse_map = PrivateGameState.getAllActionsReverseMap(curr_player_states.agent_num_cards[curr_player_states.whos_turn])
                for i, action in enumerate(new_actions):
                    if not action.is_pass:
                        action_tuple = tuple(sorted(action.idx))
                        mask[reverse_map[action_tuple]] = True
                    else:
                        mask[-1] = True
                             
                priors, net_value = self.learner.estimate_leaf_prior_value(curr_player_states, mask)
                reverse_map = PrivateGameState.getAllActionsReverseMap(curr_player_states.agent_num_cards[curr_player_states.whos_turn])
                new_priors = []
                for i, action in enumerate(new_actions):
                    if not action.is_pass:
                        action_tuple = tuple(sorted(action.idx))
                        prior = priors[reverse_map[action_tuple]]
                        new_priors.append(prior)
                    else:
                        prior = priors[-1]
                        new_priors.append(prior)
                new_priors = np.array(new_priors)
                new_priors /= new_priors.sum()
                new_nodes_val = [Tree.net_val_without_node(0, 1, 1, p, total_noises[len(curr_node.children) + i]) for i, p in enumerate(new_priors)]
                chosen_new_node = max(enumerate(new_nodes_val), key=lambda x:x[1])
                chosen_new_node_val = chosen_new_node[1]
                chosen_new_node = chosen_new_node[0]
                #print(chosen_new_node_val, chosen_child_val)
                if chosen_new_node_val < chosen_child_val:
                    curr_determined_state = curr_determined_state.getNewState(curr_node.actions[chosen_child])
                    curr_node = curr_node.children[chosen_child]
                else:
                    candidate_actions = new_actions
                    candidate_states = new_states
                    break
            
        # expansion
        leaf_node_player = curr_node.state.state.whos_turn
        if not curr_node.state.is_terminal():
             # priors, net_value = self.learner.estimate_leaf_prior_value(curr_player_states)
            # # print(priors, net_value)
            # reverse_map = PrivateGameState.getAllActionsReverseMap(curr_player_states.agent_num_cards[curr_player_states.whos_turn])
            # candidate_priors = []
            # for i, action in enumerate(candidate_actions):
            #     if not action.is_pass:
            #         action_tuple = tuple(sorted(action.idx))
            #         prior = priors[reverse_map[action_tuple]]
            #         candidate_priors.append(prior)
            #     else:
            #         prior = priors[-1]
            #         candidate_priors.append(prior)
            # candidate_priors = np.array(candidate_priors)
            # candidate_priors /= candidate_priors.sum()
            # #chosen_child = np.random.choice(range(len(candidate_actions)), p=candidate_priors)
            # chosen_child = np.argmax(candidate_priors)
            new_s = candidate_states[chosen_new_node]
            new_c = Node(new_s, curr_node)
            new_c.prior = new_priors[chosen_new_node]
            curr_node.actions.append(candidate_actions[chosen_new_node])
            curr_node.children.append(new_c)
            states = [curr_node.state.getNewState(act) for act in actions]
            available_children_mask = list(map(lambda x: check_available_children(x, states), curr_node.children))
            cached_available_nodes[-1] = available_children_mask
            curr_node = new_c
            curr_determined_state = curr_determined_state.getNewState(candidate_actions[chosen_new_node])
            # bp
            curr_available_children_idx = -1
            while curr_node != None:
                #not correct
#                print(curr_node.state.state.whos_turn, leaf_node_player)
                if curr_node.state.state.whos_turn == leaf_node_player:
                    curr_node.empirical_reward += net_value
                curr_node.play_count += 1
                #print(len(cached_available_nodes))
                curr_node = curr_node.parent
                if curr_node is not None:
                    for i, child in enumerate(curr_node.children):
                        # print("len2", len(cached_available_nodes[curr_available_children_idx]),i )
                        # print("curr", curr_available_children_idx)
                        if cached_available_nodes[curr_available_children_idx][i]:
                            child.availability_count += 1
                    curr_available_children_idx -= 1
        else:
            curr_available_children_idx = -1
            other_player = -1
            winner = curr_determined_state.who_wins()
            #it's already next turn
            for i in range(3):
                if i != curr_node.state.state.whos_turn and i != 1:
                    other_player = i
                    break
            while curr_node != None:
                if (curr_node.state.state.whos_turn == 1 and winner == 0) or (not curr_node.state.state.whos_turn == 1 and (winner == 1 or winner == 2)):
                    curr_node.empirical_reward += 1
                else:
                    curr_node.empirical_reward -= 1
                curr_node.play_count += 1
                #print(len(cached_available_nodes))
                curr_node = curr_node.parent
                if curr_node is not None:
                    for i, child in enumerate(curr_node.children):
                        # print("len2", len(cached_available_nodes[curr_available_children_idx]),i )
                        # print("curr", curr_available_children_idx)
                        if cached_available_nodes[curr_available_children_idx][i]:
                            child.availability_count += 1
                    curr_available_children_idx -= 1
            
            
