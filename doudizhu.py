import argparse

from core.agent import MctsAgent
from core.platform import Platform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", default=0)
    args = parser.parse_args()

    # initialize
    agents = [MctsAgent(i) for i in range(3)]
    for agent in agents:
        agent.start()
    # agents.insert(2, HumanAgent(2))
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
