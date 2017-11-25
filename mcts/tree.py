class Node(object):
    def __init__(self, state):
        self.children = []
        self.state = state

    def is_terminal(self):  # fix it
        pass


class InformationSet:
    def __init__(self, card_delt):
        self.card_delt = card_delt

class Tree(object):
    def __init__(self):
        self.root = Node()

    def choose(self):
        pass

    def run_iter(self):
        curr_node = self.root
        candidate_actions = []
        candidate_states = []
        while True:
            if curr_node.is_terminal():
                break
            actions = get_all_actions(curr_node, info_set)
            states = ()  # somehow acquire states
            new_states = list(filter(lambda x: x not in curr_node.children, states))
            if len(new_states) > 0:
                candidate_actions = actions
                break

        if not curr_node.is_terminal():
            pass  # choose some state
            # curr_node =
            # run until termination
