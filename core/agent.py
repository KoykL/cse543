from cards import Card, Hand

class Agent(object):

    def __init__(self):
        self.isLandlord = False

    def setLandloard(self):
        self.isLandlord = True

    def setHandcards(self, cards):
        self.handcards = cards

    def getHand(self, oldHand, isStart=False):
        isEnd = False
        if isStart:
            legalHands = self.getLegalHand(oldHand)
            hand = legalHands[0]
            return hand, isEnd
        pass

    def isWinable(self, hand, oldHand):
        return hand > oldHand
