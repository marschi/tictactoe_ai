import numpy as np

#Agent that acts randomly
class DumbAgent():
    
    def game_finished(self,score,history):
        pass
    
    def act(self,game_state):
        (config, _) = game_state
        available = np.nonzero(config.flatten() == -1)[0]
        i = np.random.choice(available)
        return (i//3,i%3)