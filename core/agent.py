import queue
import random

import numpy as np
from torch.multiprocessing import Process, Queue

import core.mcts.tree
import learning.mcts.tree
from core.cards_cython import Hand, Action
from learning.network import DeepLearner, get_model
import sys

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
                        self.t = core.mcts.tree.Tree(data)  # private_state
                else:
                    info_set = self.t.root.state
                    if event == 1:
                        info_set = info_set.getNewState(data)
                        for n in self.t.root.children:
                            if info_set == n.state:
                                print("agent {} reuse".format(self.id))
                                self.t = core.mcts.tree.Tree()
                                self.t.root = n
                                self.t.root.parent = None
                                break
                    elif event == 0 and self.t.root.state.state != data:
                        self.t = core.mcts.tree.Tree(data)
                        print("agent {} recov".format(self.id))
            if self.t is not None:
                # pr = cProfile.Profile()
                # pr.enable()
                for i in range(100):
                    # if i%100 == 0:
                    #     print("agent: ", i)
                    self.t.run_iter()
                # pr.disable()
                # sortby = 'tottime'
                # s = io.StringIO()
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()
                # print(s.getvalue())
                choice = max(self.t.root.children, key=lambda x: x.play_count)
                print("agent {} count {}".format(str(self.id), ", ".join(
                    repr(((c.play_count), str(core.mcts.tree.Tree.ucb_val(c))[:6], str(c.state.state.last_dealt_hand))) for c in
                    self.t.root.children)))
                action_idx = self.t.root.children.index(choice)
                self.decision.put((self.t.root.state.state, self.t.root.actions[action_idx]))

    def postAction(self, past_action):
        self.input_status.put((1, past_action))

    def getAction(self, private_state):
        # print(past_actions)
        counter = 0
        self.input_status.put((0, private_state))
        # actions = private_state.getLegalActions()
        # for act in actions:
        #     print(act)
        while True:
            import time
            t = time.time()
            state, decision = self.decision.get()
            print(time.time() - t)
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


class DQLAgent(BaseAgent):
    def __init__(self, id, model_path, is_training=False, turns=5):
        super().__init__(id)
        self.decision = Queue()
        self.input_status = Queue()
        self.states = Queue()
        self.t = None
        self.learner = DeepLearner(get_model(model_path))
        self.is_training = is_training
        self.turns = turns
        self.use_optimal = True
        self.actions_played = 0
    def run(self):
        while True:
            try:
                event, data = self.input_status.get(False)
                print("agent {} event {}".format(self.id, event))
            except queue.Empty:
                event = None
            else:
                if self.t is None:
                    if event == 0:
                        print("initialize tree")
                        self.t = learning.mcts.tree.Tree(self.learner, data)  # private_state
                else:
                    info_set = self.t.root.state
                    if event == 1:
                        #print("agent {} trying to reusing".format(self.id), data)
                        info_set = info_set.getNewState(data)
                        self.actions_played += 1
                        if self.actions_played >= 0:
                            self.use_optimal = True
                        for n in self.t.root.children:
                            if info_set == n.state:
                                self.t = learning.mcts.tree.Tree(self.learner)
                                self.t.root = n
                                self.t.root.parent = None
                                print("agent {} reuse".format(self.id))
                                print(self.id, "reusing", self.t.root.state.state.last_dealt_hand)
                                break
                    elif event == 0 and self.t.root.state.state != data:
                        print("agent {} recov".format(self.id))
                        print("r1", data.last_dealt_hand)
                        print("r2", self.t.root.state.state.last_dealt_hand)
                        print("r", " ".join(str(c) for c in data.agent_state.cards))
                        print("r", " ".join(str(c) for c in self.t.root.state.state.agent_state.cards))
                        self.t = learning.mcts.tree.Tree(self.learner, data)
            if self.t is not None:
                for i in range(200):
                    self.t.run_iter()
                if event is not None and event == 0:
                    if self.is_training:
                        if self.t.root.state.state.x == self.t.root.state.state.whos_turn:
                            print("using optimal: ", self.use_optimal)
                            # for i, c in enumerate(self.t.root.children):
                                # print("{}th: agent {} count {}".format(i, str(self.id), repr((c.play_count, c.empirical_reward, c.empirical_reward/c.play_count, learning.mcts.tree.Tree.net_val(c), str(c.state.state.last_dealt_hand), c.net_value, c.state.state.whos_turn, c.proped_vals))))
                                # for c in c.children:
                                #     print("because these children:", repr((c.play_count, c.empirical_reward, c.empirical_reward/c.play_count, learning.mcts.tree.Tree.net_val(c), str(c.state.state.last_dealt_hand), c.net_value, c.state.state.whos_turn, c.proped_vals)))
                                #     for c in c.children:
                                #         print("because these grandchildren:", repr((c.play_count, c.empirical_reward, c.empirical_reward/c.play_count, learning.mcts.tree.Tree.net_val(c), str(c.state.state.last_dealt_hand), c.net_value, c.state.state.whos_turn, c.proped_vals)))
                                #         for c in c.children:
                                #             print("because these grandgrandchildren:", repr((c.play_count, c.empirical_reward, c.empirical_reward/c.play_count, learning.mcts.tree.Tree.net_val(c), str(c.state.state.last_dealt_hand), c.net_value, c.state.state.whos_turn, c.proped_vals)))
                        if not self.use_optimal:
                            play_counts = np.array([c.play_count for c in self.t.root.children], dtype="float")
                            play_counts /= play_counts.sum()
                            choice = np.random.choice(self.t.root.children, p=play_counts)
                            action_idx = self.t.root.children.index(choice)
                            self.states.put(self.t.root)
                            self.decision.put((self.t.root.state.state, self.t.root.actions[action_idx]))
                        else:
                            play_counts = np.array([c.play_count for c in self.t.root.children], dtype="float")
                            play_counts /= play_counts.sum()
                            action_idx = np.argmax(play_counts)
                            self.states.put(self.t.root)
                            print("agent {} played because of these children: ".format(str(self.id)))
                            print("-".join(repr((c.play_count, c.empirical_reward, c.empirical_reward/c.play_count, learning.mcts.tree.Tree.net_val(c), str(c.state.state.last_dealt_hand))) for c in self.t.root.children[action_idx].children))
                            self.decision.put((self.t.root.state.state, self.t.root.actions[action_idx]))
                    else:
                        play_counts = np.array([c.play_count for c in self.t.root.children], dtype="float")
                        play_counts /= play_counts.sum()
                        action_idx = np.argmax(play_counts)
                        self.states.put(self.t.root)
                        self.decision.put((self.t.root.state.state, self.t.root.actions[action_idx]))
            sys.stdout.flush()
    def postAction(self, past_action):
        print("post")
        self.input_status.put((1, past_action))

    def getAction(self, private_state):
        counter = 0
        while True:
            self.input_status.put((0, private_state))
            state, decision = self.decision.get()
            if state == private_state:
                print("agent {} got {} rounds of thought".format(self.id, counter))
                counter += 1
                if counter >= self.turns:
                    break
                    # else:
                    # print("old thougts")
        print("agent {} returned after {} rounds of thought".format(self.id, counter))
        return decision

class HumanAgent(BaseAgent):
    def __init__(self, id):
        super().__init__(id)
        self.t = None
    def getAction(self, private_state):
        cards_list = sorted(private_state.agent_state.cards, key=lambda x: x.total_seq())
        print(" ".join("{}:{}".format(i, c) for i, c in enumerate(cards_list)))
        option = input("what do you want to play?")
        if option == "":
            return Action(Hand([]), True, [])
        else:
            card_indices = option.split(" ")
            action = Action(Hand(list(map(lambda x: cards_list[int(x)], card_indices))), False,
                            list(map(lambda x: int(x), card_indices)))
            action.hand.classify()
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
