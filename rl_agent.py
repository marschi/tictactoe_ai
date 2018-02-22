import numpy as np

#Agent that uses Reinforcment-Learning with Monte Carlo policy evaluation to improve
class RL_Monte_Carlo_Agent():
    
    #gamma: discount factor for future rewards
    def __init__(self, gamma=0.9, verbose=False):
        self.explore = True
        self.n_states = 2*3**9
        self.verbose = verbose
        self.value = np.zeros(self.n_states)
        self.state_visit_count = np.zeros(self.n_states,dtype=int)
        self.gamma = gamma
        self.games_played = 0

    # converts a state id to an actual board configuration and player turn
    def id_to_game_state(self,id):
      turn = id // (self.n_states // 2)
      i = id - turn * (self.n_states // 2)
      v = np.zeros(9)
      j = 8
      while i > 0:
          v[j] = i%3
          i = i // 3
          j = j - 1
      return (np.array(v).reshape((3,3)) - 1, turn)
    
    # converts a board configuration and player turn to state id
    def id_from_game_state(self,game_state):
        index = 0
        (config, turn) = game_state
        for cell in config.flatten():
            index = index * 3 + cell + 1
        index = index + turn * (self.n_states//2)
        return int(index)
    
    # callback when game is over
    # score: +1 for won game, -1 for lost game, 0 for draw
    # history: complete history of the game from player perspective
    def game_finished(self,score,history):
        # just for interest
        self.games_played = self.games_played + 1
        if not self.explore: return
        reward = score
        # t = steps into the past
        for t,actual_state in enumerate(history):
            for state in self.get_similar_states(actual_state):
                i = self.id_from_game_state(state)
                # compute 'future' value that we got from being in that state
                game_reward = reward * self.gamma ** t
                self.state_visit_count[i] = self.state_visit_count[i] + 1
                self.alpha = 1.0/self.state_visit_count[i]
                self.value[i] = self.value[i] + self.alpha * (game_reward - self.value[i])
    
    #get the value that we get from performing action in config
    def get_action_value(self,action,config):
        possible_config = np.copy(config)
        possible_config[action] = 0
        return self.value[self.id_from_game_state((possible_config, 0))]

    def get_action_exploration_status(self, action, config):
        possible_config = np.copy(config)
        possible_config[action] = 0
        return self.state_visit_count[self.id_from_game_state((possible_config, 0))]

    #given a config, find the possible action that yields the best value
    def get_best_option(self,config):
        available = np.nonzero(config.flatten() == -1)[0]
        available = [(action//3,action%3) for action in available]
        best_option = np.argmax([self.get_action_value(action, config) for action in available])
        return available[best_option]

    #given a config, find the possible action that leads to the least explored state
    def get_least_explored_option(self,config):
        available = np.nonzero(config.flatten() == -1)[0]
        available = [(action//3,action%3) for action in available]
        explore_status = [self.get_action_exploration_status(action, config) for action in available]
        least_explored = np.argmin(explore_status)
        return available[least_explored]

    #For exploiting the symmetry and rotation of the game
    def get_similar_states(self,state):
        config, turn = state
        config = np.copy(config)
        similar = []
        for i in range(3):
            similar.append((config,turn))
            similar.append((np.flip(config,axis=0),turn))
            similar.append((np.flip(config,axis=1),turn))
            config = np.rot90(config)
        return similar
        
    
    #find out the coordinates of the field that we want to occupy this turn
    def act(self,game_state):
        if self.explore:
            (config, _) = game_state
            return self.get_least_explored_option(config)
        else:
          (config, _) = game_state
          best_option = self.get_best_option(config)
          if (self.verbose):
              print(game_state)
              print(best_option)
          return best_option