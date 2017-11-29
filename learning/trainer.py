import os
import pickle


class GameHistory(object):
    def __init__(self, trees):
        self.path = []
        self.data = []
        paths = [tree.path for tree in trees]
        # Not finished yet


def playOnce(platform, verbose=False):
    # start game
    platform.deal()
    if verbose:
        for i, a_s in enumerate(platform.game_state.agent_states):
            print("agent {} has card: {}".format(i, a_s.get_cards_str()))
    while not platform.game_state.isTerminal():
        agent_playing = platform.game_state.whos_turn
        action = platform.turn()
        if verbose:
            print("agent {} played: {}".format(agent_playing, action))
            for i, a_s in enumerate(platform.game_state.agent_states):
                print("agent {} has card: {}".format(i, a_s.get_cards_str()))
    # save GameHistory
    history = GameHistory([agent.t for agent in agents])
    output_path = 'data/history/'
    # pickle.dump(history)
    # end game
    for agent in agents:
        agent.terminate()
