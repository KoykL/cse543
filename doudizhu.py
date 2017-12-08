import argparse
import logging
import os
from itertools import count
from multiprocessing import set_start_method

from core.agent import MctsAgent, HumanAgent, DQLAgent
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
    parser.add_argument("--human_player", default=None, action="store", type=int)
    args = parser.parse_args()
    logger = logging.getLogger(None)
    if args.train:
        output_path = 'data/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        trainer = DQLTrainer(os.path.join(output_path, "model.pth"), os.path.join(output_path, "optimizer.pth"),
                             os.path.join(output_path, "memory.pkl"))
        for i in count():
            print("DQLTrainer: run an iteration")
            print("running iteration", i)
            trainer.run_iter()
    else:
        # initialize
        ai_nums = list(range(0, 3))
        if args.human_player is not None:
            ai_nums.remove(args.human_player)
        
        agents = [DQLAgent(i, "data/model.pth", turns=15) for i in ai_nums]
        
        for agent in agents:
            agent.start()
        if args.human_player is not None:
            agents.insert(args.human_player, HumanAgent(args.human_player))
        # agents.insert(2, RandomAgent())
        platform = Platform(agents)

        # start game
        platform.deal()
        if args.human_player is None:
            for i, a_s in enumerate(platform.game_state.agent_states):
                print("agent {} has card: {}".format(i, a_s.get_cards_str()))
            else:
                print("you have card: {}".format(platform.game_state.agent_states[args.human_player]))
        while not platform.game_state.isTerminal():
            agent_playing = platform.game_state.whos_turn
            action = platform.turn()
            print("agent {} played: {}".format(agent_playing, action))
            if args.human_player is None:
                for i, a_s in enumerate(platform.game_state.agent_states):
                    print("agent {} has card: {}".format(i, a_s.get_cards_str()))
            else:
                print("you have card: {}".format(platform.game_state.agent_states[args.human_player]))

        for agent in agents:
            agent.terminate()
