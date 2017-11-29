import argparse
import os

from core.agent import MctsAgent, HumanAgent
from core.platform import Platform
from learning.trainer import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", default=0)
    parser.add_argument("-t", "--train", default=0, type=int)
    args = parser.parse_args()

    if args.train > 0:
        output_path = 'data/history/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        # initialize
        agents = [MctsAgent(i) for i in range(3)]
        for agent in agents:
            agent.start()
        platform = Platform(agents)
        playOnce(platform, True)
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
