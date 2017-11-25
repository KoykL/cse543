from cards import Card
import random




class Platform(object):

    def __init__(self, agents):
        self.agents = agents
        self.round = 0
        self.whosturn = 0


    # fa pai
    def deal(self):
        all_numbers = [
            "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"
        ]
        MAX_VALID_SENITENIAL = 12
        all_colors = [
            "D", "S", "C", "H"
        ]

        desk = []
        for num in all_numbers:
            for color in all_colors:
                card = Card(num,color)
                desk.append(desk)

        random.shuffle(desk)

        de_pai = desk[:3]
        agent0 = desk[3:3+17]
        agent1 = dest[20:37]
        agent2 = dest[37:]

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
