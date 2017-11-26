from core.mcts.tree import Tree
class Agent(object):

    def __init__(self):
        self.t = None

    #     self.isLandlord = False
    #
    # def setLandlord(self):
    #     self.isLandlord = True
    #
    # @property
    # def cards(self):
    #     return self.cards
    #
    # @cards.setter
    # def cards(self, cards):
    #     self.cards = sorted(cards, key=lambda x: x.seq())

    def getAction(self, private_state, past_actions):
        if self.t is None:
            self.t = Tree(private_state)
        else:
            info_set = self.t.root.state
            for act in past_actions:
                print("a")
                info_set = info_set.getNewState(act)
                some_reuse = False
                for n in self.t.root.children:
                    if info_set == n.state:
                        print("reuse")
                        self.t = Tree()
                        self.t.root = n
                        self.t.root.parent = None
                        some_reuse = True
                        break
                if not some_reuse:
                    print("recov")
                    self.t = Tree(private_state)
                    break
        for i in range(1000):
            if i%100 == 0:
                print("agent: ", i)
            self.t.run_iter()
        choice = max(self.t.root.children, key=lambda x: x.empirical_reward / x.play_count)
        action_idx = self.t.root.children.index(choice)
        return self.t.root.actions[action_idx]
    # def isWinnable(self, hand):
    #     return self.cards == hand
