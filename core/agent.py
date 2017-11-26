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

    # def isWinnable(self, hand):
    #     return self.cards == hand
