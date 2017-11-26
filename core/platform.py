import random
from contextlib import suppress
from copy import deepcopy
from itertools import combinations

from core.cards import Card, Hand, NotComparableError


class Platform(object):

    def __init__(self, agents):
        self.agents = agents
        agent_states = [AgentState([]) for _ in agents]
        self.game_state = GameState.new(agent_states)

    @staticmethod
    def get_deck():
        deck = []
        for num in Card.all_numbers:
            for color in Card.all_colors:
                card = Card(num, color)
                deck.append(card)
        return deck

    # fa pai
    def deal(self):
        self.game_state.agent_states[0].setLandlord()

        deck = Platform.get_deck()
        random.shuffle(deck)

        di_pai = deck[:3]
        self.game_state.agent_states[0].cards.update(deck[:3+17])
        self.game_state.agent_states[1].cards.update(deck[20:37])
        self.game_state.agent_states[2].cards.update(deck[37:])

    # da pai
    def turn(self):
        if self.game_state.isTerminal():
            raise RuntimeError("Game has reached terminal state")
        private_state = self.game_state.getPrivateStateForAgentX(self.game_state.who_wins)
        action = self.agents[self.game_state.who_wins].getAction(private_state)
        self.game_state = self.game_state.getNewState(action)



# I need
# 1. getallactions(gamestate)
# 2. getnewstate(action)
# 3. check if a state is terminal state
# 4. at terminal state, check tell who won
# All methods should be as fast as possible. If needed, change the API

#implement gamestate equality check
class Action(object):
    def __init__(self, hand, is_pass):
        self.is_pass = is_pass
        self.hand = hand

class PrivateGameState(object):
    def __init__(self, x, game_state):
        self.x = x
        self.agent_state = game_state.agent_state
        self.pass_count = game_state.pass_count
        self.whos_turn = game_state.whos_turn
        self.last_dealt_hand = game_state.last_dealt_hand
        self.dealt_cards = game_state.dealt_cards
        self.agent_num_cards = [len(h) for h in game_state.agent_states.cards]

    def getPublicInstantiation(self, others_instantiation):
        agents = list(others_instantiation)
        agents.insert(self.x, self.agent_state)
        return GameState(agents, self.pass_count, self.whos_turn, self.last_dealt_hand, self.dealt_cards)

    def getNextActions(self):
        cards_set = self.agent_states[self.whos_turn].cards
        cards = sorted(cards_set, key=lambda x: x.seq())
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
                pairs.append([card, last_card])
                # Triplet
                if last_card.cmp_number(last_last_card) == 0:
                    triplets.append([card, last_card, last_last_card])
                    # boom
                    if last_last_card.cmp_number(last_last_last_card) == 0:
                        bombs.append([card, last_card, last_last_card, last_last_last_card])
            last_card = card
            last_last_card = last_card
            last_last_last_card = last_last_card
        two_triplets = list(map(lambda x: x[0] + x[1], combinations(triplets, 2)))

        # straight

        # Han
        # nums = sorted(set(map(lambda x:x.number, cards)), key=lambda x:Card.all_numbers.index(x))
        # for start in range(len(nums) - 4):
        #     if Card.all_numbers.index(nums[start + 4]) == Card.all_numbers.index(nums[start]) + 4:
        #         straight_nums.append(nums[start:start + 5]
        # for straight_num in straight_nums:
        #     for c in cards:
        #         if card.number == straight_num[0]:
        #             c1s.append(c)
        #         if card.number == straight_num[1]:
        #             c2s.append(c)
        #         if card.number == straight_num[2]:
        #             c3s.append(c)
        #         if card.number == straight_num[3]:
        #             c4s.append(c)
        #         if card.number == straight_num[4]:
        #             c5s.append(c)
        #     for c1 in c1s:
        #         for c2 in c2s:
        #             for c3 in c3s:
        #                 for c4 in c4s:
        #                     for c5 in c5s:
        #                         hand = Hand([c1, c2, c3, c4, c5])
        #                         actions.append(hand)

        # Charlie
        # for i in range(8):
        #     for j in range(4):
        #         for k in range(4):
        #             for l in range(4):
        #                 for m in range(4):
        #                     for n in range(4):
        #                         s = set([j, k, l, m, n])
        #                         if(len(s) != 1):
        #                             continue
        #                         else:
        #                             straight = set()
        #                             straight.add(Card(Card.all_numbers[i], Card.all_colors[j]))
        #                             straight.add(Card.all_numbers[i+1], Card.all_colors[k])
        #                             straight.add(Card.all_numbers[i+2], Card.all_colors[l])
        #                             straight.add(Card.all_numbers[i+3], Card.all_colors[m])
        #                             straight.add(Card.all_numbers[i+4], Card.all_colors[n])
        #                             if straight.issubset(cards_set):
        #                                 actions.append(straight)

        # Johnny
        index = 0
        straight = []
        cur_num = -1
        last_num = -1
        for card in cards:
            if index == 0:
                straight = [card]
            cur_num = card.number

            if cur_num == last_num + 1:
                index = index + 1
                straight.append(card)
            elif cur_num == last_num:
                pass
            else:
                index = 0
                straight = []  # reinitialize

            if index >= 4:
                actions.append(straight[index - 4:index + 1])

            last_num = cur_num

        # three pairs
        index = 0
        straight = []
        cur_num = -1
        last_num = -1
        for pair in pairs:
            if index == 0:
                straight = pair
            cur_num = pair[0].number

            if cur_num == last_num + 1:
                index = index + 1
                straight = straight + pair
            elif cur_num == last_num:
                pass
            else:
                index = 0
                straight = []  # reinitialize

            if index >= 2:
                actions.append(straight[2 * index - 5:2 * index + 1])

            last_num = cur_num

        # nuke
        bj = Card("BJ", "")
        rj = Card("RJ", "")
        if bj in cards_set and rj in cards_set:
            actions.append(Hand([bj, rj]))

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
                if pair[0].cmp_number(bomb[0]) != 0:
                    # four with 1 pair
                    # hand = Hand(bomb + pair)
                    # actions.append(hand)

                    # four with two pairs
                    for another_pair in pairs:
                        if another_pair[0].cmp_number(pair[0]) != 0 and another_pair[0].cmp_number(bomb[0]) != 0:
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
                        if len(another_pair) == 2 and not another_pair[0].cmp_number(pair[0]) and not another_pair[
                            0].cmp_number(two_triplet[0]):
                            hand = Hand(two_triplet + pair + another_pair)
                            actions.append(hand)

        actions = [Action(act, False) for act in actions]
        return actions

    def getLegalActions(self):
        actions = self.getNextActions()
        if self.last_dealt_hand is None:
            return actions
        else:
            next_actions = []
            for hand in actions:
                with suppress(NotComparableError):
                    if hand > self.last_dealt_hand:
                        next_actions.append(hand)
            next_actions.append(Action([], True))
        return next_actions


class GameState(object):
    def __init__(self, agent_states, pass_count, whos_turn, last_dealt_hand, dealt_cards):
        self.agent_states = agent_states
        self.pass_count = pass_count
        self.whos_turn = whos_turn
        self.last_dealt_hand = last_dealt_hand
        self.dealt_cards = dealt_cards

    @staticmethod
    def new(agent_states):
        # agent_states = []
        # for i in range(3):
        # agent_states.append(AgentState(cards))
        return GameState(agent_states, 0, 0, None, set())

    def isTerminal(self):
        for agent_s in self.agent_states:
            if len(agent_s.cards) == 0:
                return True
        return False

    def getNewState(self, action):
        new_agent_states = deepcopy(self.agent_states)
        new_agent_states[self.whos_turn] = new_agent_states[self.whos_turn].do_deal_cards(action)
        new_dealt_hand = action if not action.is_pass else self.last_dealt_hand
        new_pass_count = self.pass_count + 1 if action.is_pass else self.pass_count
        if new_pass_count == 2:
            new_pass_count = 0
            new_dealt_hand = None
        true_action = action if not action.is_pass else []
        state = GameState(new_agent_states,
                          pass_count=new_pass_count,
                          whos_turn=(self.whos_turn + 1) % 3,
                          last_dealt_hand=new_dealt_hand,
                          dealt_cards=self.dealt_cards | true_action)
        return state

    def who_wins(self):
        if not self.isTerminal():
            raise RuntimeError("not yet terminal state")
        else:
            for i, agent_s in enumerate(self.agent_states):
                if len(agent_s.cards) == 0:
                    return i

    def getPrivateStateForAgentX(self, x):
        return PrivateGameState(self, x)

class AgentState(object):
    def __init__(self, cards, isLandlord=False):
        self.cards = set(cards) #set()
        self.isLandlord = isLandlord
        #self.last_dealt_hand =
    def do_deal_cards(self, hand):
        return AgentState(self.cards - set(hand))
    def setLandlord(self):
        self.isLandlord = True
