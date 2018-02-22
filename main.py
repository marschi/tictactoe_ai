from game import TicTacToeGame
from rl_agent import RL_Monte_Carlo_Agent
from dumb_agent import DumbAgent
import numpy as np

agent = RL_Monte_Carlo_Agent()
# Play a couple games against itself to explore the game
for _ in range(1000):
    game = TicTacToeGame(agent,agent)
    while not game.game_over:
        game.play_turn()
# Now use the obtained knowledge to optmize behaviour
agent.explore = False

# Lets see how it does against a dumb player
dumbAgent = DumbAgent()
wins = 0
draws = 0
losses = 0
for _ in range(1000):
    game = TicTacToeGame(dumbAgent,agent)
    while not game.game_over:
        game.play_turn()
    if game.winner == 1:
        wins = wins + 1
    elif game.winner == 0:
        losses = losses + 1
    else:
        draws = draws + 1
    
print('WINS: ',wins,', DRAWS: ',draws,', LOSSES: ',losses)
print('WIN/DRAW-RATE: ', (wins + draws) / (wins + draws + losses))
