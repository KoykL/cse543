import math
import random
from copy import deepcopy

from core.platform import AgentState


class Node(object):
    def __init__(self, info_set, parent):
        self.actions = []
        self.children = []
        self.parent = parent
        self.state = info_set

        self.empirical_reward = 0
        self.play_count = 1

        # self.availability_count = 1
        self.priors = 0


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
    def ucb_val(node):
        k = 1
        return (node.empirical_reward / node.play_count) + k * math.sqrt(
            math.log(node.prior) / node.play_count)

    def run_iter(self):
        curr_node = self.root
        if self.determinization is None or self.iterations_per_determinization >= 250:
            curr_determined_state = curr_node.state.determinization()
            self.determinization = deepcopy(curr_determined_state)
            self.iterations_per_determinization = 0
            # print("change determinization")
        else:
            curr_determined_state = deepcopy(self.determinization)
            self.iterations_per_determinization += 1

        candidate_actions = []
        candidate_states = []
        candidate_determined_states = []

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
            # print("hand:", curr_determined_state.getPrivateStateForAgentX(curr_determined_state.whos_turn).agent_state.get_cards_str())
            # print("agent num:", curr_determined_state.whos_turn)
            # print("\n".join(str(act) for act in actions))
            determined_states = [curr_determined_state.getNewState(act) for act in actions]
            states = [curr_node.state.getNewState(act) for act in actions]
            available_children_mask = list(map(lambda x: check_available_children(x, states), curr_node.children))
            # info_states = [InformationSet(s) for s in states]
            states_mask = list(map(lambda x: not check_in_children(x, curr_node), states))
            new_states = list(map(lambda c: c[1], filter(lambda c: states_mask[c[0]], enumerate(states))))
            new_actions = list(map(lambda c: c[1], filter(lambda c: states_mask[c[0]], enumerate(actions))))
            if len(new_states) > 0:
                candidate_actions = new_actions
                candidate_states = new_states
                candidate_determined_states = determined_states
                break
            else:
                # print("children len", len(curr_node.children), len(actions))
                children_vals = [(i, Tree.ucb_val(c)) for i, c in
                                 filter(lambda c: available_children_mask[c[0]], enumerate(curr_node.children))]
                chosen_child = max(children_vals, key=lambda x: x[1])
                chosen_child = chosen_child[0]
                curr_determined_state = curr_determined_state.getNewState(curr_node.actions[chosen_child])
                curr_node = curr_node.children[chosen_child]

        # expansion
        if not curr_node.state.is_terminal():
            # chosen_child = random.randint(0, len(candidate_actions) - 1)
            # print("fix this")
            priors, value = self.learner.estimate_leaf_prior_value(curr_node.state)

            # TODO: fix this
            raise NotImplementedError("fix here")

            # all_cards_indices = PrivateGameState.getAllActions(len(curr_node.state.agent_state.cards))

            # all_prior_indices = zip(priors, all_cards_indices)
            all_usable_prior_actions = []
            # for prior, indices in all_prior_indices:
            #     if indices[0] < 0:
            #         continue
            # action = []
            # for idx in indices:
            # action.append(curr_node.state.agent_state.cards[idx])
            # fix last action: pass
            # all_usable_prior_actions.append((prior, Action(Hand(action), False)))

            # all_usable_prior_actions.append((priors[-1], Action(Hand([]), True)))


            for i, action in enumerate(zip(candidate_actions)):
                new_s = candidate_states[i]
                new_c = Node(new_s, curr_node)
                curr_node.actions.append(action)
                curr_node.children.append(new_c)
                curr_node = new_c
                curr_determined_state = candidate_determined_states[i]
                # TODO: fix this
        # run until termination
        # print("winner", winner)
        # backpropagation
        # print(cached_available_nodes)
        while curr_node != None:
            other_player = -1
            for i in range(3):
                if i != curr_node.state.state.whos_turn and i != 0:
                    other_player = i
                    break
            #
            if (curr_node.state.state.agent_state.isLandlord and winner == curr_node.state.state.whos_turn) or (
                        not curr_node.state.state.agent_state.isLandlord and (
                                    winner == curr_node.state.state.whos_turn or winner == other_player)):
                # print("reward once")
                curr_node.empirical_reward += 1

            curr_node.play_count += 1
            curr_node = curr_node.parent