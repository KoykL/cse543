import random
from itertools import combinations

from core.cards import Card
from agent import Agent
from copy import deepcopy

class Platform(object):

    def __init__(self, agents):
        self.agents = agents
        self.round = 0
        self.whos_turn = 0
        self.startGameState = GameState(agents)

    @staticmethod
    def get_deck():
        deck = []
        for num in Card.all_numbers:
            for color in Card.all_colors:
                card = Card(num,color)
                deck.append(desk)
        return deck

    # fa pai
    def deal(self):

        agents[0].setLandlord

        deck = Platform.get_deck()
        random.shuffle(desk)

        di_pai = deck[:3]
        agents[0].setHandcards = deck[3:3+17]
        agents[1].setHandcards = deck[20:37]
        agents[2].setHandcards = deck[37:]

    # da pai
    def run(self):
        hand, isEnd = self.agents[0].getAction(hand, isStart)
        pass

    def end_game(self):
        pass


# I need
# 1. getallactions(gamestate)
# 2. getnewstate(action)
# 3. check if a state is terminal state
# 4. at terminal state, check tell who won
# All methods should be as fast as possible. If needed, change the API

#implement gamestate equality check
class GameState(object):
    def __init__(self, agent_states, pass_count, whos_turn, last_dealt_hand, dealt_cards): #
        self.agent_states = agent_states
        self.pass_count = pass_count
        self.whos_turn = whos_turn
        self.last_dealt_hand = last_dealt_hand #Hand()
        self.dealt_cards = dealt_cards#set()
        self.who_wins = None

    @staticmethod
    def new(agent_states):
        # agent_states = []
        # for i in range(3):
            # agent_states.append(AgentState(cards))
        return GameState(agent_states, 0, 0, None, set())

    def getLegalActions(self):
        cards = self.agent_states[self.whos_turn].cards

        # single card
        actions = [Hand([card]) for card in cards]
        pairs, triplets, bombs = [], [], []
        index = 0
        last_card = None
        last_last_card = None
        last_last_last_card = None
        for card in cards:
            # pairs
            if card.cmp_number(last_card) == 0:
                pairs.append([card,last_card])
                #Triplet
                if last_card.cmp_number(last_last_card) == 0:
                    triplets.append([card,last_card, last_last_card])
                    #boom
                    if last_last_card.cmp_number(last_last_last_card) == 0:
                        bombs.append([card,last_card, last_last_card,last_last_last_card])
            last_card = card
            last_last_card = last_card
            last_last_last_card = last_last_card
        two_triplets = list(map(lambda x:x[0]+x[1], combinations(triplets, 2)))

        # straight
        straight_nums = []
        nums =["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        for start in range(len(nums) - 4):
            if Card.all_numbers.index(nums[start + 4]) == Card.all_numbers.index(nums[start]) + 4:
                straight_nums.append(nums[start:start + 5]
        for straight_num in straight_nums:
            for c1 in [card for card in cards if card.number == straight_num[0]]:
                for c2 in [card for card in cards if card.number == straight_num[1]]:
                    for c3 in [card for card in cards if card.number == straight_num[2]]:
                        for c4 in [card for card in cards if card.number == straight_num[3]]:
                            for c5 in [card for card in cards if card.number == straight_num[4]]:
                                hand = Hand([c1, c2, c3, c4, c5])
                                actions.append(hand)

        # nuke
        for card in cards:
            if card.number == "BJ":
                for another_card in cards:
                    if another_card.number == "RJ":
                        hand = Hand([card]+[another_card])
                        actions.append(hand)
        # kickers
        for triplet in triplets:
            # 3 with 1
            for card in cards:
                if card not in triplet:
                    hand = Hand(triplet + [card])
                    actions.append(hand)
            # 3 with 2
            for pair in pairs:
                if not pair[0].cmp_number(triplet[0]):
                    hand = Hand(triplet + [pair])
                    actions.append(hand)

        for bomb in bombs:
            # four with 1
            for card in cards:
                if card not in bomb:
                    hand = Hand(bomb + [card])
                    actions.append(hand)

                    # four with 2 singles
                    for another_card in cards:
                        if another_card != card and another_card not in bomb:
                            hand = Hand(bomb + [card] + [another_card])
                            actions.append(hand)

            for pair in pairs:
                if not pair[0].cmp_number(bomb[0]):
                    # four with 1 pair
                    hand = Hand(bomb + pair)
                    actions.append(hand)

                    #four with two pairs
                    for another_pair in pairs:
                        if not another_pair[0].cmp_number(pair[0]) and not another_pair[0].cmp_number(bomb[0]):
                            hand = Hand(bomb + pair + another_pair)
                            actions.append(hand)

                    # four with 1 and 2
                    for card in cards:
                        if card not in bomb and card not in pair:
                            hand = Hand(bomb + pair + [card])
                            actions.append(hand)

        for two_triplet in two_triplets:
            # 2 triplets with 2 singles
            for card in cards:
                if card not in two_triplet:
                    for another_card in cards:
                        if another_card != card and another_card not in two_triplets:
                            hand = Hand(two_triplet + [card] + [another_card])
                            actions.append(hand)

            # 2 triplets with 2 pairs
            for pair in pairs:
                if len(pair) == 2 and not pair[0].cmp_number(two_triplet[0]):
                    for another_pair in pairs:
                        if len(another_pair) == 2 and not another_pair[0].cmp_number(pair[0]) and not another_pair[0].cmp_number(two_triplet[0]):
                            hand = Hand(two_triplet + pair + another_pair)
                            actions.append(hand)
        return actions
    def isTerminal(self):
        for agent_s in self.agent_states:
            if len(agent_s.cards) == 0:
                return True
        return False

    def getNewState(self, action):
        new_agent_states = deepcopy(self.agent_states)
        new_agent_states[self.whos_turn] = new_agent_states[self.whos_turn].after_dealt_cards(action)
        state = GameState(agent_states, pass_count, whos_turn, last_dealt_hand, dealt_cards)
        return state

    def who_wins(self):
        if not self.isTerminal():
            raise RuntimeError("not yet terminal state")
        else:
            for i, agent_s in enumerate(self.agent_states):
                if len(agent_s.cards) == 0:
                    return i
    def getPrivateStateForAgentX(x):
        return PrivateGameState(self, x)


class PrivateGameState(object):
    def __init__(self, game_state, x):
        self.x = x
        self.agent_state = game_state.agent_states[x]
        self.pass_count = game_state.pass_count
        self.whos_turn = game_state.whos_turn
        self.last_dealt_hand = game_state.last_dealt_hand
        self.dealt_cards = game_state.delta_cards

        self.agent_num_cards = [len(h) for h in game_state.agent_states.cards]

    def getPublicInstantiation(self, others_instantiation):
        agents = list(others_instantiation)
        agents.insert(x, self.agent_state)
        return GameState(agents, self.pass_count, self.whos_turn, self.last_dealt_hand, self.dealt_cards)

class AgentState(object):
    def __init__(self, cards):
        self.cards = cards #set()
        #self.last_dealt_hand =
    def after_dealt_cards(self, hand):
        return self.cards - set(hand)
