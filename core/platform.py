import functools
import itertools
import random
from contextlib import suppress
from copy import copy
from itertools import combinations

from core.cards_cython import Card, Hand, Action, NotComparableError, get_action_cwrapper


class Platform(object):

    def __init__(self, agents):
        assert len(agents) == 3, "number of agents should be 3"
        self.agents = agents

        agent_states = [AgentState([]) for _ in agents]
        self.game_state = GameState.new(agent_states)
        self.round = 0
        self.actions = []
    @staticmethod
    @functools.lru_cache(None)
    def get_deck():
        deck = []
        for num in Card.all_numbers[:-2]:
            for color in Card.all_colors:
                card = Card(num, color)
                deck.append(card)
        bj = Card("BJ", "")
        rj = Card("RJ", "")
        deck.append(bj)
        deck.append(rj)
        return deck

    # fa pai
    def deal(self):
#        self.game_state.agent_states[0].setLandlord()

        deck = Platform.get_deck()
        random.shuffle(deck)

        di_pai = deck[:3]
        self.game_state.agent_states[0].append_hand(deck[:3 + 17])
        self.game_state.agent_states[1].append_hand(deck[20:37])
        self.game_state.agent_states[2].append_hand(deck[37:])

    # da pai
    def turn(self):
        if self.game_state.isTerminal():
            raise RuntimeError("Game has reached terminal state")
#        assert self.game_state.agent_states[0].isLandlord
        private_state = self.game_state.getPrivateStateForAgentX(self.game_state.whos_turn)
        action = self.agents[self.game_state.whos_turn].getAction(private_state)
        self.game_state = self.game_state.getNewState(action)
        self.round += 1
        self.actions.append(action)
        for agent in self.agents:
            agent.postAction(action)
        return action

# I need
# 1. getallactions(gamestate)
# 2. getnewstate(action)
# 3. check if a state is terminal state
# 4. at terminal state, check tell who won
# All methods should be as fast as possible. If needed, change the API

#implement gamestate equality check
class PrivateGameState(object):
    def __init__(self, x, agent_state, pass_count, whos_turn, last_dealt_hand, dealt_cards, agent_num_cards, other_cards):
        self.x = x
        self.agent_state = agent_state
        self.pass_count = pass_count
        self.whos_turn = whos_turn
        self.last_dealt_hand = last_dealt_hand
        self.dealt_cards = dealt_cards
        self.agent_num_cards = agent_num_cards
        self.other_cards = other_cards

    def __eq__(self, other):
        # print('private comp')
        return self.x == other.x and self.agent_state == other.agent_state and self.pass_count == other.pass_count and self.whos_turn == other.whos_turn and self.last_dealt_hand == other.last_dealt_hand and self.dealt_cards == other.dealt_cards and self.agent_num_cards == other.agent_num_cards and self.other_cards == other.other_cards
    @staticmethod
    def from_game_state(x, game_state):
        all_cards = set(Platform.get_deck())
        others_cards = all_cards - set(game_state.agent_states[x].cards) - game_state.dealt_cards
        return PrivateGameState(x, game_state.agent_states[x], game_state.pass_count, game_state.whos_turn, game_state.last_dealt_hand, game_state.dealt_cards, [len(h.cards) for h in game_state.agent_states], others_cards)

    def getPublicInstantiation(self, others_instantiation):
        agents = list(others_instantiation)
        agents.insert(self.x, self.agent_state)
        return GameState(agents, self.pass_count, self.whos_turn, self.last_dealt_hand, self.dealt_cards)

    @staticmethod
    @functools.lru_cache(None)
    def max_combinations(max_length=20):
        possibilities = []
        l1 = list(itertools.combinations(range(max_length), 1))
        l2 = list(itertools.combinations(range(max_length), 2))
        l3 = list(itertools.combinations(range(max_length), 3))
        l4 = list(itertools.combinations(range(max_length), 4))
        l5 = list(itertools.combinations(range(max_length), 5))
        l6 = list(itertools.combinations(range(max_length), 6))
        return len(l1), len(l2), len(l3), len(l4), len(l5), len(l6)

    @staticmethod
    @functools.lru_cache(None)
    def getAllActions(length, max_length=20):
        l1, l2, l3, l4, l5, l6 = PrivateGameState.max_combinations(max_length)
        possibilities = [-1] * (l1 + l2 + l3 + l4 + l5 + l6)
        possibilities[0:l1] = list(itertools.combinations(range(length), 1))
        possibilities[l1:l1 + l2] = list(itertools.combinations(range(length), 2))
        possibilities[l1 + l2:l1 + l2 + l3] = list(itertools.combinations(range(length), 3))
        possibilities[l1 + l2 + l3:l1 + l2 + l3 + l4] = list(itertools.combinations(range(length), 4))
        possibilities[l1 + l2 + l3 + l4:l1 + l2 + l3 + l4 + l5] = list(itertools.combinations(range(length), 5))
        possibilities[l1 + l2 + l3 + l4 + l5:l1 + l2 + l3 + l4 + l5 + l6] = list(
            itertools.combinations(range(length), 6))
        return possibilities

    @staticmethod
    @functools.lru_cache(None)
    def getAllActionsReverseMap(length, max_length=20):
        pos = {}
        actions = PrivateGameState.getAllActions(length)
        for i, p in enumerate(actions):
            pos[p] = i
        if -1 in pos:
            del pos[-1]
        return pos
    def getNextActions(self):
        cards = self.agent_state.cards
        # return PrivateGameState.getNextActionsStatic(cards)
        return PrivateGameState.get_action_wrapper(cards)

    @staticmethod
    @functools.lru_cache(10000)
    def get_action_wrapper(cards):
        return get_action_cwrapper(cards)

    @staticmethod
    @functools.lru_cache(10000)
    def getNextActionsStatic(cards):
        # single card
        # numbers_set = set()
        actions = []
        for i, card in enumerate(cards):
            actions.append(Action(Hand([card]), False, [i]))
            # numbers_set.add(card.number)

        set_of_number = []
        set_of_current_number = []
        for i, card in enumerate(cards):
            if len(set_of_current_number) == 0:
                set_of_current_number.append((i, card))
            elif card.number == set_of_current_number[0][1].number:
                set_of_current_number.append((i, card))
            else:
                set_of_number.append(set_of_current_number)
                set_of_current_number = [(i, card)]
        set_of_number.append(set_of_current_number)
        pairs, triplets, bombs = [], [], []
        for set_of_current_number in set_of_number:
            if len(set_of_current_number) > 1:
                for p in list(combinations(set_of_current_number, 2)):
                    pair = []
                    for i, c in p:
                        pair.append((i, c))
                    indices, hand_pair = zip(*pair)
                    pairs.append(Action(Hand(hand_pair), False, indices))
            if len(set_of_current_number) > 2:
                for t in list(combinations(set_of_current_number, 3)):
                    triplet = []
                    for i, c in t:
                        triplet.append((i, c))
                    indices, hand_triplet = zip(*triplet)
                    triplets.append(Action(Hand(hand_triplet), False, indices))
            if len(set_of_current_number) > 3:
                indices, hand_bomb = zip(*set_of_current_number)
                bombs.append(Action(Hand(hand_bomb), False, indices))
        # pairs, triplets, bombs = [], [], []
        # index = 0
        # last_card = None
        # last_card_index = -1
        # last_last_card = None
        # last_last_card_index = -1
        # last_last_last_card = None
        # last_last_last_card_index = -1
        # for i, card in enumerate(cards):
        #     # pairs
        #     if last_card is not None and card.cmp_number(last_card) == 0 :
        #         pairs.append(Action(Hand([card, last_card]), False, [i, last_card_index]))
        #         # Triplet
        #         if last_last_card is not None and last_card.cmp_number(last_last_card) == 0:
        #             triplets.append(Action(Hand([card, last_card, last_last_card]), False, [i, last_card_index, last_last_card_index]))
        #             # boom
        #             if last_last_last_card is not None and last_last_card.cmp_number(last_last_last_card) == 0:
        #                 bombs.append(Action(Hand([card, last_card, last_last_card, last_last_last_card]), False, [i, last_card_index, last_last_card_index, last_last_last_card_index]))
        #
        #     last_last_last_card_index = last_last_card_index
        #     last_last_last_card = last_last_card
        #     last_last_card_index = last_card_index
        #     last_last_card = last_card
        #     last_card_index = i
        #     last_card = card

        two_triplets = []
        for triplet in triplets:
            for another_triplet in triplets:
                if triplet.hand[0].seq() == another_triplet.hand[0].seq() - 1 and another_triplet.hand[0].number != "2":
                    two_triplets.append(
                        Action(triplet.hand + another_triplet.hand, False, triplet.idx + another_triplet.idx))

        three_pairs = []
        for pair1 in pairs:
            for pair2 in pairs:
                for pair3 in pairs:
                    if pair1.hand[0].seq() == pair2.hand[0].seq() - 1 and pair2.hand[0].seq() == pair3.hand[
                        0].seq() - 1 and pair3.hand[0].number != "2":
                        three_pairs.append(
                            Action(pair1.hand + pair2.hand + pair3.hand, False, pair1.idx + pair2.idx + pair3.idx))

        actions += pairs + triplets + bombs + two_triplets + three_pairs
        # two_triplets = list(map(lambda x: x[0] + x[1], combinations(triplets, 2)))

        # straight

        # Han
        # nums = sorted(set(map(lambda x:x.number, cards)), key=lambda x:Card.all_numbers.index(x))
        # for start in range(len(nums) - 4):
        #     if Card.all_numbers.index(nums[start + 4]) == Card.all_numbers.index(nums[start]) + 4:
        #         straight_nums.append(nums[start:start + 5])
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
        # length = 0
        straight_actions = []
        straight = []
        straight_indices = []
        cur_num = -10
        last_num = -10
        suppress_new_cards = 0
        for i, card in enumerate(cards):
            if card.seq() < Card.MAX_VALID_SENITENIAL:
                cur_num = card.seq()
                if len(straight) == 0:
                    suppress_new_cards = 0
                    straight = [[card]]
                    straight_indices = [[i]]
                if cur_num == last_num + 1:
                    suppress_new_cards = 0
                    for s, ids in zip(straight, straight_indices):
                        s.append(card)
                        ids.append(i)
                        if len(s) >= 5:
                            straight_actions.append(Action(Hand(s[len(s) - 5: len(s)]), False, ids[len(s) - 5: len(s)]))
                elif cur_num == last_num:
                    new = []
                    new_ids = []
                    for s, ids in zip(straight, straight_indices):
                        new.append(copy(s))
                        new_ids.append(copy(ids))
                    new_new = []
                    new_new_ids = []
                    for n, ns in zip(list(new[:-1 - suppress_new_cards] + [new[-1]]),
                                     new_ids[:-1 - suppress_new_cards] + [new_ids[-1]]):
                        n[-1] = card
                        ns[-1] = i
                        if len(n) >= 5:
                            straight_actions.append(Action(Hand(n[len(n) - 5: len(n)]), False, ns[len(n) - 5: len(n)]))

                        new_new.append(n)
                        new_new_ids.append(ns)
                        suppress_new_cards += 1
                    straight.extend(new_new)
                    straight_indices.extend(new_new_ids)

                else:
                    suppress_new_cards = 0
                    straight = [[card]]  # reinitialize
                    straight_indices = [[i]]
                last_num = cur_num

        real_straight_actions = []
        for i, action1 in enumerate(straight_actions):
            unique = True
            for j, action2 in enumerate(straight_actions):
                if j == i or j < i:
                    continue
                if action1 == action2:
                    unique = False
            if unique:
                real_straight_actions.append(action1)
        # for a in straight_actions: print(a)
        # for a in real_straight_actions: print(a)
        actions.extend(real_straight_actions)
        # three pairs (by Kengdie Johnny)
        # three_pair = []
        # cur_num = -10
        # last_num = -10
        # for pair in pairs:
        #     if pair.hand[0].seq() < Card.MAX_VALID_SENITENIAL:
        #         if len(three_pair) == 0:
        #             three_pair = [pair]
        #             length = 1
        #         cur_num = pair.hand[0].seq()
        #         if cur_num == last_num + 1:
        #             for tp in three_pair:
        #                 tp += pair
        #                 print(tp)
        #                 if len(tp.hand) >= 6:
        #                     actions.append(Action(tp.hand[2 * length - 6:2 * length], False, tp.idx[2 * length - 6:2 * length]))
        #         elif cur_num == last_num:
        #             new = []
        #             for tp in three_pair:
        #                 new.append(Action(copy(tp.hand), False, copy(tp.idx)))
        #             new_new = []
        #             for n in list(new):
        #                 n_hand = copy(n.hand)
        #                 n_idx = copy(n.idx)
        #                 n_hand[-2:] = pair.hand
        #                 n_idx[-2:] = pair.idx
        #                 n = Action(n_hand, False, n_idx)
        #                 if len(n.hand) >= 6:
        #                     actions.append(Action(n.hand[-6: len(n.hand)], False, n.idx[-6: len(n.hand)]))
        #                 new_new.append(n)
        #             three_pair.extend(new_new)
        #         else:
        #             three_pair = [pair]  # reinitialize
        #         last_num = cur_num

        # nuke
        bj = Card("BJ", "")
        rj = Card("RJ", "")
        with suppress(ValueError):
            i = cards.index(bj)
            j = cards.index(rj)
            actions.append(Action(Hand([bj, rj]), False, [i, j]))

        # kickers
            # for triplet in triplets:
            #     # 3 with 1
            #     for card in cards:
            #         if card not in triplet:
            #             hand = Hand(triplet + [card])
            #             actions.append(hand)
            #     # 3 with 2
            #     for pair in pairs:
            #         if pair[0].cmp_number(triplet[0]):
            #             hand = Hand(triplet + pair)
            #             actions.append(hand)

            # for bomb in bombs:
            #     # four with 1
            #     for card in cards:
            #         if card not in bomb:
            #             hand = Hand(bomb + [card])
            #             actions.append(hand)
            #
            #             # four with 2 singles
            #             for another_card in cards:
            #                 if another_card != card and another_card not in bomb:
            #                     hand = Hand(bomb + [card] + [another_card])
            #                     actions.append(hand)
            #
            #     for pair in pairs:
            #         if pair[0].cmp_number(bomb[0]):
            #             # four with 1 pair
            #             # hand = Hand(bomb + pair)
            #             # actions.append(hand)
            #
            #             # four with two pairs
            #             for another_pair in pairs:
            #                 if another_pair[0].cmp_number(pair[0]) and another_pair[0].cmp_number(bomb[0]):
            #                     hand = Hand(bomb + pair + another_pair)
            #                     actions.append(hand)
            #
            #             # four with 1 and 2
            #             for card in cards:
            #                 if card.cmp_number(bomb[0]) and card.cmp_number(pair[0]):
            #                     hand = Hand(bomb + pair + [card])
            #                     actions.append(hand)

            # for two_triplet in two_triplets:
            #     # 2 triplets with 2 singles
            #     for card in cards:
            #         if card.cmp_number(two_triplet[0]) and card.cmp_number(two_triplet[-1]):
            #             for another_card in cards:
            #                 if another_card.cmp_number(card) and another_card.cmp_number(two_triplet[0]) and another_card.cmp_number(two_triplet[-1]):
            #                     hand = Hand(two_triplet + [card] + [another_card])
            #                     actions.append(hand)
            #
            #     # 2 triplets with 2 pairs
            #     for pair in pairs:
            #         if pair[0].cmp_number(two_triplet[0]) and pair[0].cmp_number(two_triplet[-1]):
            #             for another_pair in pairs:
            #                 if another_pair[0].cmp_number(pair[0]) and another_pair[0].cmp_number(two_triplet[0]) and another_pair[0].cmp_number(two_triplet[-1]):
            #                     hand = Hand(two_triplet + pair + another_pair)
            #                     actions.append(hand)
        # print("actions", len(actions))
        # for act in actions:
        #     print(act)
        # print("exit")
        # exit(-1)
        # actions = [Action(act, False) for act in actions]
        for act in actions:
            act.hand.classify()
        for act in actions:
            assert act.hand.type != "invalid", act
        # print("branches: ", len(actions))
        return actions

    def getLegalActions(self):
        actions = self.getNextActions()
        if self.last_dealt_hand is None:
            return actions
        else:
            next_actions = []
            for action in actions:
                # print("legal hand1", " ".join(str(c) for c in action.hand), action.hand.type)
                # print("legal hand2", " ".join(str(c) for c in self.last_dealt_hand.hand), action.hand.type)

                # print( str(action))
                # print( str(self.last_dealt_hand))
                with suppress(NotComparableError):
                    if action.hand > self.last_dealt_hand.hand:
                        next_actions.append(action)
            pass_act = Action(Hand([]), True, [])
            pass_act.hand.classify()
            next_actions.append(pass_act)
        return next_actions

    def isTerminal(self):
        for cards in self.agent_num_cards:
            # print("cards: ", cards)
            assert cards >= 0, cards
            if cards == 0:
                return True
        return False

    def getNewState(self, action, who):
        assert action.hand.type != "unclassified", action.hand.type
        true_action = action.hand if not action.is_pass else []
        new_agent_state = self.agent_state
        new_others_cards = self.other_cards
        if who == self.x:
            new_agent_state = new_agent_state.do_deal_cards(action)
        else:
            new_others_cards = self.other_cards - set(true_action)
            if len(new_others_cards) != len(self.other_cards) - len(true_action):
                raise RuntimeError("invalid hand: {}, with my hand: {}".format(str(action), " ".join(
                    str(card) for card in sorted(self.other_cards, key=lambda x: x.seq()))))
        new_dealt_hand = action if not action.is_pass else self.last_dealt_hand
        new_pass_count = self.pass_count + 1 if action.is_pass else 0
        if new_pass_count == 2:
            new_pass_count = 0
            new_dealt_hand = None
        new_agent_num_cards = copy(self.agent_num_cards)
        if(len(action.hand) > new_agent_num_cards[who]):
            print("error throw more card than I have")
            print(action)
            print(who)
            exit(-1)
        # print(new_agent_num_cards[who], len(action.hand))
        new_agent_num_cards[who] -= len(action.hand)

        return PrivateGameState(
            x=self.x,
            agent_state=new_agent_state,
            pass_count=new_pass_count,
            whos_turn=(self.whos_turn + 1) % 3,
            last_dealt_hand=new_dealt_hand,
            dealt_cards=self.dealt_cards|set(true_action),
            agent_num_cards=new_agent_num_cards,
            other_cards=new_others_cards)


class GameState(object):
    def __init__(self, agent_states, pass_count, whos_turn, last_dealt_hand, dealt_cards):
        self.agent_states = agent_states
        self.pass_count = pass_count
        self.whos_turn = whos_turn
        self.last_dealt_hand = last_dealt_hand
        self.dealt_cards = dealt_cards

    def __eq__(self, other):
        return self.agent_states == other.agent_states and self.pass_count == other.pass_count and self.whos_turn == other.whos_turn and self.last_dealt_hand == other.last_dealt_hand and self.dealt_cards == other.dealt_cards

    @staticmethod
    def new(agent_states):
        # agent_states = []
        # for i in range(3):
        # agent_states.append(AgentState(cards))
        return GameState(agent_states, 0, 0, None, set())

    def getNewState(self, action):
        assert action.hand.type != "unclassified", action.hand.type
        new_agent_states = copy(self.agent_states)
        new_agent_states[self.whos_turn] = new_agent_states[self.whos_turn].do_deal_cards(action)
        new_dealt_hand = action if not action.is_pass else self.last_dealt_hand
        new_pass_count = self.pass_count + 1 if action.is_pass else 0
        if new_pass_count == 2:
            new_pass_count = 0
            new_dealt_hand = None
        true_action = action.hand if not action.is_pass else []
        state = GameState(new_agent_states,
                          pass_count=new_pass_count,
                          whos_turn=(self.whos_turn + 1) % 3,
                          last_dealt_hand=new_dealt_hand,
                          dealt_cards=self.dealt_cards | set(true_action))
        return state

    def isTerminal(self):
        for agent_s in self.agent_states:
            if len(agent_s.cards) == 0:
                return True
        return False

    def who_wins(self):
        if not self.isTerminal():
            raise RuntimeError("not yet terminal state")
        else:
            for i, agent_s in enumerate(self.agent_states):
                if len(agent_s.cards) == 0:
                    return i

    def getPrivateStateForAgentX(self, x):
        return PrivateGameState.from_game_state(x, self)


class AgentState(object):
    def __init__(self, cards):
        self.cards = tuple(sorted(cards, key=lambda x: x.total_seq()))  # list()
        #self.last_dealt_hand =

    def __eq__(self, other):
        if self.cards.__hash__ == other.cards.__hash__:
            return True
        return self.cards == other.cards

    def do_deal_cards(self, hand):
        if not hand.is_pass:
            result = AgentState(set(self.cards) - set(hand.hand))
            if len(result.cards) != len(self.cards) - len(hand.hand):
                raise RuntimeError("invalid hand: {}, with my hand: {}".format(hand, self.get_cards_str()))
            return result
        else:
            return self

    def append_hand(self, cards):
        new_cards = self.cards + tuple(cards)
        self.cards = tuple(sorted(new_cards, key=lambda x: x.total_seq()))
    # def setLandlord(self):
    #     self.isLandlord = True

    def get_cards_str(self):
        return " ".join(map(str,sorted(self.cards, key=lambda x:x.total_seq())))
