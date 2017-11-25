import collections


class Card(object):
    all_numbers = [
        "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"
    ]
    all_colors = [
        "D", "S", "C", "H"
    ]

    def __init__(self, color, number):
        self.color = color
        self.number = number

    def is_joker(self):
        return self.number == "BJ" or self.number == "RJ"

    def __eq__(self, other):
        return self.number == other.number and self.color == other.color

    def cmp_number(self, other):
        if self.number == other.number:
            return 0
        elif Card.all_numbers.index(self.number) < Card.all_numbers.index(other.number):
            return -1
        else:
            return 1


class Hand(list):
    def __init__(self, it):
        self.extend(it)

    def classify(self):
        if len(self) == 1:
            self.type = "single"
        elif len(self) == 2:
            if self[0] == self[1]:
                self.type = "pair"
            elif self[0].is_joker() and self[1].is_joker():
                self.type = "bomb"
            else:
                self.type = "invalid"
        elif len(self) == 3:
            if self[0] == self[1] and self[1] == self[2]:
                self.type = "triple"
            else:
                self.type = "invalid"
        elif len(self) == 4:
            counter = collections.Counter(self)
            counts = set(counter.values())
            pass

    def __cmp__(self, other):
        pass


class CardDeck(list):
    def __init__(self):
        super().__init__(self)
