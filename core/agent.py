import random

from core.cards import Hand
from core.mcts.tree import Tree
from core.platform import Action


class BaseAgent(object):

    def __init__(self):
        pass


class MctsAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.t = None

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

    def getAction(self, private_state, past_actions):
        # print(past_actions)
        if self.t is None:
            self.t = Tree(private_state)
        else:
            info_set = self.t.root.state
            for act in past_actions:
                print("a")
                info_set = info_set.getNewState(act)
                some_reuse = False
                for n in self.t.root.children:
                    if info_set == n.state:
                        print("reuse")
                        self.t = Tree()
                        self.t.root = n
                        self.t.root.parent = None
                        some_reuse = True
                        break
                if not some_reuse:
                    print("recov")
                    self.t = Tree(private_state)
                    break
        for i in range(1000):
            # print("agent", i)
            if i%100 == 0:
                print("agent: ", i)
            self.t.run_iter()
        choice = max(self.t.root.children, key=lambda x: x.empirical_reward / x.play_count)
        action_idx = self.t.root.children.index(choice)
        return self.t.root.actions[action_idx]
    # def isWinnable(self, hand):
    #     return self.cards == hand


class HumanAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.t = None

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

    def getAction(self, private_state, past_actions):
        cards_list = sorted(private_state.agent_state.cards, key=lambda x: x.seq())
        print(" ".join("{}:{}".format(i, c) for i, c in enumerate(cards_list)))
        option = input("what do you want to play?")
        if option == "":
            return Action(Hand([]), True)
        else:
            card_indices = option.split(" ")
            action = Action(Hand(list(map(lambda x: cards_list[int(x)], card_indices))), False)
            print("you played: {}".format(action))
            return action


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

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

    def getAction(self, private_state, past_actions):
        actions = private_state.getLegalActions()
        return random.choice(actions)
