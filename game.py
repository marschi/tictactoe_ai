import numpy as np

class InvalidMoveException(Exception):
    pass

class GameOverException(Exception):
    pass

class TicTacToeGame():
    def __init__(self, agent1, agent2):
        # a random player will have the first turn
        self.turn = np.random.randint(2)
        self.agents = [agent1, agent2]
        # starting with an empty board configuration obviously
        self.config = -np.ones((3,3))
        self.game_over = False
        self.winner = None
        # number of turns made in that game
        self.n_turns = 0
        # complete history of the game (from newest to oldest)
        self.history = [(self.config, self.turn)]

    #plays one turn of the game
    def play_turn(self):
        if self.game_over:
            raise GameOverException('Cannot play anymore. Game is over!')
        # Swap player turn
        self.turn = 1 - self.turn
        player_config = self.config_from_player_perspective(self.config,self.turn)
        # let the player make his turn
        row, col = self.agents[self.turn].act((player_config,self.turn))
        if self.config[row,col] != -1:
            raise ValueError('Invalid Action!')
        self.config = np.copy(self.config)
        self.config[row,col] = self.turn
        # add resulting config to history
        self.history = [(self.config, self.turn)] + self.history
        self.n_turns = self.n_turns + 1
        # Game over
        if self.n_turns >= 9 or self.has_won(self.turn):
            # history from the players perspectives
            player_hist = np.array([None,None])
            player_hist[0] = np.array(self.history)
            player_hist[1] = np.array([(self.config_from_player_perspective(state,1),1-player) for state, player in self.history])
            if self.n_turns >= 9:
                #thats a draw
                self.agents[self.turn].game_finished(0, player_hist[self.turn])
                self.agents[1 - self.turn].game_finished(0, player_hist[1 - self.turn])
            if self.has_won(self.turn):
                #the player won with his last turn
                self.agents[self.turn].game_finished(1, player_hist[self.turn])
                self.agents[1 - self.turn].game_finished(-1, player_hist[1 - self.turn])
                self.winner = self.turn
            self.game_over = True
    
    # gives the configuration from players perspective, where itself will always be player 0
    def config_from_player_perspective(self,config, player):
        return (config == 1 - player) * 1 + (config == -1) * -1
    
    # wether a player has won the game with the current config
    def has_won(self,player):
        pos = self.config == player
        return np.any(pos.sum(axis=0) == 3) or \
            np.any(pos.sum(axis=1) == 3) or \
            np.all(np.diag(pos)) or \
            np.all(np.diag(np.fliplr(pos)))

