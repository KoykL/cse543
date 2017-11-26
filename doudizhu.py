import argparse
from core.platform import Platform
from core.agent import Agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", default=0)
    args = parser.parse_args()

    # initialize
    agents = [Agent() for i in range(3)]
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
