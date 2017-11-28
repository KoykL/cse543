import queue
import random
from multiprocessing import Process, Queue

from core.cards import Hand
from core.mcts.tree import Tree
from core.platform import Action


class BaseAgent(Process):
    def __init__(self, id):
        super().__init__()
        self.id = id

    def postAction(self, past_action):
        pass

    def getAction(self, private_state):
        raise NotImplementedError()
class MctsAgent(BaseAgent):
    def __init__(self, id):
        super().__init__(id)
        self.t = None
        self.decision = Queue()
        self.input_status = Queue()
        self.init = True
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
    def run(self):
        while True:
            try:
                event, data = self.input_status.get(False)
            except queue.Empty as e:
                pass
            else:
                if self.t is None:
                    if event == 0:
                        print("initialize tree")
                        self.t = Tree(data)  # private_state
                else:
                    info_set = self.t.root.state
                    if event == 1:
                        info_set = info_set.getNewState(data)
                        for n in self.t.root.children:
                            if info_set == n.state:
                                print("agent {} reuse".format(self.id))
                                self.t = Tree()
                                self.t.root = n
                                self.t.root.parent = None
                                break
                    elif event == 0 and self.t.root.state.state != data:
                        self.t = Tree(data)
                        print("agent {} recov".format(self.id))
            if self.t is not None:
                for i in range(100):
                    # if i%100 == 0:
                    #     print("agent: ", i)
                    self.t.run_iter()
                choice = max(self.t.root.children, key=lambda x: x.empirical_reward / x.play_count)
                action_idx = self.t.root.children.index(choice)
                self.decision.put((self.t.root.state.state, self.t.root.actions[action_idx]))

    def postAction(self, past_action):
        self.input_status.put((1, past_action))

    def getAction(self, private_state):
        # print(past_actions)
        counter = 0
        self.input_status.put((0, private_state))
        actions = private_state.getLegalActions()
        for act in actions:
            print(act)
        while True:
            state, decision = self.decision.get()
            if state == private_state:
                print("agent {} got {} rounds of thought".format(self.id, counter))
                counter += 1
                if self.init:
                    if counter > 100:
                        self.init = False
                        break
                else:
                    if counter > 60:
                        break
                        # else:
                        # print("old thougts")
        print("agent {} returned after {} rounds of thought".format(self.id, counter))
        return decision

    # def isWinnable(self, hand):
    #     return self.cards == hand


class HumanAgent(BaseAgent):
    def __init__(self, id):
        super().__init__(id)
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
    def getAction(self, private_state):
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
    def __init__(self, id):
        super().__init__(id)

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

    def getAction(self, private_state):
        actions = private_state.getLegalActions()
        return random.choice(actions)
