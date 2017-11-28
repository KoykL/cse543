import collections
from functools import total_ordering

class Card(object):
    all_numbers = [
        "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"
    ]
    MAX_VALID_SENITENIAL = 12
    all_colors = [
        "D", "S", "C", "H"
    ]

    def __init__(self, number, color):
        self.color = color
        self.number = number

    def is_joker(self):
        return self.number == "BJ" or self.number == "RJ"
    def __eq__(self, other):
        return self.number == other.number and self.color == other.color

    def __hash__(self):
        return (self.number + self.color).__hash__()

    def seq(self):
        return Card.all_numbers.index(self.number)

    def color_seq(self):
        return Card.all_colors.index(self.color)
    @staticmethod
    def static_seq(num):
        return Card.all_numbers.index(num)

    def cmp_number(self, other):
        if self.number == other.number:
            return 0
        elif self.seq() < other.seq():
            return -1
        else:
            return 1

    def __str__(self):
        return "[{},{}]".format(self.color, self.number)

class NotComparableError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@total_ordering
class Hand(list):
    def __init__(self, it):
        super().__init__(it)
        self.sort(key=lambda x: x.seq())
        self.classify()
    def __eq__(self, other):
        if self.type != other.type:
            return False
        else:
            return self.cmp(other) == 0

    def __lt__(self, other):
        return self.cmp(other) == -1

    def __ne__(self, other):
        if self.type != other.type:
            return True

        else:
            return self.cmp(other) != 0

    def __le__(self, other):
        return self.cmp(other) <= 0

    def __gt__(self, other):
        return self.cmp(other) > 0

    def __ge__(self, other):
        return self.cmp(other) >= 0

    def classify(self):
        counter = collections.Counter([n.number for n in self])
        numbers = set(counter.keys())
        seqs = [Card.static_seq(n) for n in numbers]
        counts = set(counter.values())
        sorted_seqs = list(sorted(seqs))
        if len(self) == 1:
            self.type = "single"
        elif len(self) == 2:
            if self[0].seq() == self[1].seq():
                self.type = "pair"
            elif self[0].is_joker() and self[1].is_joker():
                self.type = "bomb"
            else:
                self.type = "invalid"
        elif len(self) == 3:
            if self[0].seq() == self[1].seq() and self[1].seq() == self[2].seq():
                self.type = "triple"
            else:
                self.type = "invalid"
        elif len(self) == 4:
            if 3 in counts and 1 in counts:
                self.type = "threewithone"
            elif 4 in counts:
                self.type = "bomb"
            else:
                self.type = "invalid"
        elif len(self) == 5:
            if 3 in counts and 2 in counts:
                self.type = "threewithtwo"
            elif 4 in counts and 1 in counts:
                self.type = "fourwithone"
            elif 1 in counts and sorted_seqs[0] + 4 == sorted_seqs[-1] and sorted_seqs[-1] < Card.MAX_VALID_SENITENIAL:
                self.type = "straight"
            else:
                self.type = "invalid"
        elif len(self) == 6:
            if 2 in counts and len(counts) == 1 and (
                        sorted_seqs[0] + 2 == sorted_seqs[1] + 1 and sorted_seqs[1] + 1 == sorted_seqs[2]) and \
                            sorted_seqs[2] < Card.MAX_VALID_SENITENIAL:
                self.type = "threepairs"
            elif 3 in counts and len(counts) == 1 and sorted_seqs[0] + 2 == sorted_seqs[1] + 1 and sorted_seqs[
                1] < Card.MAX_VALID_SENITENIAL:
                self.type = "twotriple"
            elif 4 in counts and len(counts) == 2:
                self.type = "fourwithtwo"
            else:
                self.type = "invalid"
        elif len(self) == 7:
            if 4 in counts and (2 in counts and 1 in counts) or (3 in counts):
                self.type = "fourwithpairwithsingle"
            else:
                self.type = "invalid"
        elif len(self) == 8:
            if 4 in counts and 2 in counts and len(counts) == 2:
                self.type = "fourwithtwopairs"
            elif (3 in counts and (1 in counts or 2 in counts) and len(counts) == 2)  :
                two_triple = list(filter(lambda x: x[1] == 3, counter.items()))
                triple_seq = [Card.static_seq(n[0]) for n in two_triple]
                triple_sorted_seqs = list(sorted(triple_seq))
                if triple_sorted_seqs[0] + 1 == triple_sorted_seqs[1] and triple_sorted_seqs[
                    1] < Card.MAX_VALID_SENITENIAL:
                    self.type = "twotriplewithtwosingles"
                else:
                    self.type = "invalid"
            elif (4 in counts and 3 in counts and 1 in counts):
                triple = list(filter(lambda x: x[1] == 3, counter.items()))[0][0]
                quad = list(filter(lambda x: x[1] == 4, counter.items()))[0][0]
                triple_seq = Card.static_seq(triple)
                quad_seq = Card.static_seq(quad)
                if triple_seq == quad_seq + 1 or triple_seq + 1 == quad_seq:
                    self.type = "twotriplewithtwosingles"
                else:
                    self.type = "invalid"
            else:
                self.type = "invalid"
        elif len(self) == 10:
            if 3 in counts and 2 in counts and len(counts) == 2:
                two_triple = list(map(lambda x: x[0], filter(lambda x: x[1] == 3, counter.items())))
                triple_seq = [Card.static_seq(n) for n in two_triple]
                triple_sorted_seqs = list(sorted(triple_seq))
                if triple_sorted_seqs[0] + 1 == triple_sorted_seqs[1] and triple_sorted_seqs[
                    1] < Card.MAX_VALID_SENITENIAL:
                    self.type = "twotriplewithtwopairs"
                else:
                    self.type = "invalid"
            else:
                self.type = "invalid"
        else:
            self.type = "invalid"

    def cmp(self, other):
        if self.type == "invalid" or other.type == "invalid":
            raise NotImplementedError("cannot compare invalid hands")
        elif self.type == other.type and self.type != "bomb":
            counter1 = collections.Counter([n.number for n in self])
            counter2 = collections.Counter([n.number for n in other])
            numbers1 = set(counter1.keys())
            seqs1 = [Card.static_seq(n) for n in numbers1]
            numbers2 = set(counter2.keys())
            seqs2 = [Card.static_seq(n) for n in numbers2]
            sorted_seqs1 = list(sorted(seqs1))
            sorted_seqs2 = list(sorted(seqs2))
            if self.type == "single" or self.type == "pair" or self.type == "triple":
                return self[0].cmp_number(other[0])
            elif self.type == "threewithone" or self.type == "threewithtwo" or self.type == "twotriplewithtwosingles" or self.type == "twotriplewithtwopairs":
                triple_part1 = set(map(lambda x: x[0],filter(lambda x: x[1] == 3, counter1.items())))
                triple_cards1 = list(
                    sorted(filter(lambda x: x.number in triple_part1, self), key=lambda x: x.seq()))

                triple_part2 = set(map(lambda x: x[0],filter(lambda x: x[1] == 3, counter2.items())))
                triple_cards2 = list(
                    sorted(filter(lambda x: x.number in triple_part2, other), key=lambda x: x.seq()))
                return triple_cards1[0].cmp_number(triple_cards2[0])
            elif self.type == "fourwithone" or self.type == "fourwithtwo" or self.type == "fourwithpairwithsingle" or self.type == "fourwithtwopairs":
                triple_part1 = set(map(lambda x: x[0], filter(lambda x: x[1] == 4, counter1.items())))
                triple_cards1 = list(
                    sorted(filter(lambda x: x.number in triple_part1, self), key=lambda x: x.seq()))

                triple_part2 = set(map(lambda x: x[0], filter(lambda x: x[1] == 4, counter2.items())))
                triple_cards2 = list(
                    sorted(filter(lambda x: x.number in triple_part2, other), key=lambda x: x.seq()))
                return triple_cards1[0].cmp_number(triple_cards2[0])
            elif self.type == "threepairs" or self.type == "twotriple" or self.type == "straight":
                if sorted_seqs1[0] == sorted_seqs2[0]:
                    return 0
                elif sorted_seqs1[0] < sorted_seqs2[0]:
                    return -1
                else:
                    return 1
            else:
                raise NotComparableError("self type: {}, other type: {}".format(self.type, other.type))
        else:
            if self.type == "bomb":
                return 1
            elif other.type == "bomb":
                return -1
            else:
                raise NotComparableError("self type: {}, other type: {}".format(self.type, other.type))

class CardDeck(list):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    cards1 = set([Card("BJ", "D"), Card("RJ", "D")])
    hand1 = Hand(cards1)
    cards2 = set([Card("A", "D"), Card("A", "S"), Card("A", "C"), Card("A", "H")])
    hand2 = Hand(cards2)
    print(hand1.type)
    print(hand2.type)
    print(hand1 > hand2)
