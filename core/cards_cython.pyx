# distutils: language=c++
# cython: profile=True

#remove later
import collections
import functools
from libcpp.map cimport map as cppmap
from libcpp.vector cimport vector

cdef class Card(object):
    all_numbers = [
        "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"
    ]
    MAX_VALID_SENITENIAL = 12
    all_colors = [
        "D", "S", "C", "H"
    ]

    numbers = {"3":0, "4":1, "5":2, "6":3, "7":4, "8":5, "9":6, "10":7, "J":8, "Q":9, "K":10, "A":11, "2":12, "BJ":13, "RJ":14}
    colors = {"D":0, "S":1, "C":2, "H":3}

    cdef int idx
    cdef public str number
    cdef public str color

    def __init__(self, number, color):
        if number == "BJ":
            self.idx = 52
        elif number == "RJ":
            self.idx = 53
        else:
            self.idx = Card.numbers[number] * 4 + Card.colors[color]
        self.number = number
        self.color = color

    cpdef get_idx(self):
        return self.idx

    cpdef is_joker(self):
        # return self.number == "BJ" or self.number == "RJ"
        return self.idx > 51

    def __eq__(self, Card other):
        return self.idx == other.idx

    def __hash__(self):
        # return (self.number + self.color).__hash__()
        return self.idx.__hash__()

    cpdef seq(self):
        # return Card.all_numbers.index(self.number)
        if self.idx == 52: return 13
        if self.idx == 53: return 14
        return self.idx // 4

    cpdef color_seq(self):
        # return Card.all_colors.index(self.color)
        return self.idx % 4

    cpdef total_seq(self):
        seq = self.seq()
        if seq < 13:
            color_seq = self.color_seq()
            return seq * 4 + color_seq
        else:
            return 52 + seq - 13
        # return self.idx

    @staticmethod
    def static_seq(num):
        # return Card.all_numbers.index(num)
        return Card.numbers[num]

    cpdef cmp_number(self, Card other):
        # if self.number == other.number:
        #     return 0
        # elif self.seq() < other.seq():
        #     return -1
        # else:
        #     return 1
        if self.seq() == other.seq():
            return 0
        elif self.seq() < other.seq():
            return -1
        else:
            return 1

    def __str__(self):
        # return "[{},{}]".format(self.color, self.number)
        if self.idx == 52: return "[BJ]"
        if self.idx == 53: return "[RJ]"
        return "[{}{}]".format(Card.all_colors[self.idx % 4], Card.all_numbers[self.idx // 4])

class NotComparableError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

cdef extern from "cfunc.h":
    int get_class_c(int count, cppmap[int, int] h)
    vector[vector[int]] get_action_c(vector[int])

class Hand(tuple):

    def __new__(cls, it):
        return tuple.__new__(cls, tuple(sorted(it, key=lambda x: x.seq())))

    def __init__(self, it):
        super().__init__()
        self.classify()

    def __add__(self, it):
        res = super().__add__(it)
        return Hand(sorted(res, key=lambda x: x.seq()))

    def __eq__(self, other):
        return super().__eq__(other)

    def __ne__(self, other):
        if self.type != other.type:
            return True
        else:
            return self.cmp(other) != 0

    def __lt__(self, other):
        return self.cmp(other) == -1

    def __le__(self, other):
        return self.cmp(other) <= 0

    def __gt__(self, other):
        return self.cmp(other) > 0

    def __ge__(self, other):
        return self.cmp(other) >= 0

    def __hash__(self):
        return super().__hash__()

    @functools.lru_cache(10000)
    def get_class(self):
        cdef cppmap[int, int] hand
        cdef int count = len(self)
        for card in self:
            hand[translate(card.get_idx())] += 1
        type = get_class_c(count, hand)
        return type

    def classify(self):
        self.value = self.get_class()
        self.type = self.value // 100

    def cmp(self, other):
        Invalid = 0 #未知
        Single = 100 #单张
        Double = 200 #对子
        Triplet = 300 #三条
        Straight = 400 #单顺
        ThreePairs = 500 #双顺
        TwoTriplet = 600 #双顺
        Bomb = 700 #炸弹
        if self.type == 0 or other.type == 0:
            raise NotImplementedError("cannot compare invalid hands")
        elif self.type == other.type and self.type != Bomb:
            if self.value == other.value:
                return 0
            elif self.value < other.value:
                return -1
            else:
                return 1
        elif self.type == other.type and self.type == "bomb":
            if self.value == other.value:
                return 0
            elif self.value < other.value:
                return -1
            else:
                return 1
        else:
            if self.type == "bomb":
                return 1
            elif other.type == "bomb":
                return -1
            else:
                raise NotComparableError("self {} type: {}, other {} type: {}".format(''.join([str(c) for c in self]), self.value, ''.join([str(c) for c in other]), other.value))

cdef int translate(int num):
    if (num < 52):
        return num / 4 + 3
    else:
        return num - 36

cpdef get_action_cwrapper(cs):
    cdef vector[int] cards
    ids = {}
    actions = []
    for i, c in enumerate(cs):
        idx = c.get_idx()
        cards.push_back(c.get_idx())
        ids[idx] = i
    hands = get_action_c(cards)
    for hand in hands:
        indices = [ids[card] for card in hand]
        handcards = [cs[id] for id in indices]
        action = Action(Hand(handcards), False, indices)
        actions.append(action)
    for act in actions:
        assert act.hand.type != 0, act
    return actions

class Action(object):
    def __init__(self, hand, is_pass, idx):
        self.is_pass = is_pass
        # if type(idx) == type(tuple()):
        #     raise RuntimeError("who the fuck passed a tuple?")
        self.idx = list(idx)
        # assert(type(hand) == type(Hand))
        self.hand = Hand(hand)

    def __eq__(self, other):
        return type(self) == type(other) and self.is_pass == other.is_pass and self.hand == other.hand
    def __str__(self):
        # print(self.idx)
        if self.is_pass:
            return "[pass]"
        else:
            return "{}".format(" ".join(repr((i, str(card))) for i, card in zip(self.idx, self.hand)))

    def __add__(self, other):
        return Action(self.hand + other.hand, self.is_pass and other.is_pass, self.idx + other.idx)

class CardDeck(list):
    def __init__(self):
        super().__init__()
