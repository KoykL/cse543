import argparse
import logging
import os
from itertools import count
from multiprocessing import set_start_method

from core.agent import MctsAgent, HumanAgent
from core.platform import Platform
from learning.trainer import DQLTrainer

try:
    set_start_method('spawn')
except RuntimeError:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-a", "--agent", default=0)
    parser.add_argument("-t", "--train", default=False, action='store_true')
    args = parser.parse_args()
    logger = logging.getLogger(None)
    if args.train:
        output_path = 'data/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        trainer = DQLTrainer(os.path.join(output_path, "model.pth"))
        for i in count():
            print("DQLTrainer: run an iteration")
            print("running iteration", i)
            trainer.run_iter()
    else:
        # initialize
        agents = [MctsAgent(i) for i in range(1, 3)]
        for agent in agents:
            agent.start()
        agents.insert(0, HumanAgent(0))
        # agents.insert(2, RandomAgent())
        platform = Platform(agents)

        # start game
        platform.deal()
        for i, a_s in enumerate(platform.game_state.agent_states):
            print("agent {} has card: {}".format(i, a_s.get_cards_str()))

        while not platform.game_state.isTerminal():
            agent_playing = platform.game_state.whos_turn
            action = platform.turn()
            print("agent {} played: {}".format(agent_playing, action))
            for i, a_s in enumerate(platform.game_state.agent_states):
                print("agent {} has card: {}".format(i, a_s.get_cards_str()))
        for agent in agents:
            agent.terminate()
