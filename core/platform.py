import random

from core.cards import Card


class Platform(object):

    def __init__(self, agents):
        self.agents = agents
        self.round = 0
        self.whosturn = 0


    # fa pai
    def deal(self):
        desk = []
        for num in Card.all_numbers:
            for color in Card.all_colors:
                card = Card(num,color)
                desk.append(desk)

        random.shuffle(desk)

        de_pai = desk[:3]
        agent0 = desk[3:3+17]
        agent1 = desk[20:37]
        agent2 = desk[37:]

        pass

    # da pai
    def run(self):
        hand, isEnd = self.agents[0].getAction(hand, isStart)
        pass

    def end_game(self):
        pass


#
class GameState:
    def __init__(self, agents):
        self.agents = agents
        self.round = 0
        self.whosturn = 0
