from tree import Tree
class Agent(object):

    def __init__(self):
        pass

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

    def getAction(self, private_state):
        t = Tree(private_state)
        for i in range(1000):
            t.run_iter()
        choice = max(t.root.children, key=lambda x: x.empirical_reward / x.play_count)
        action_idx = t.root.children.index(choice)
        return t.root.actions[action_idx]
    # def isWinnable(self, hand):
    #     return self.cards == hand
