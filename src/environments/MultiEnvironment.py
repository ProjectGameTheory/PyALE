class MultiEnvironment(object):

    def __init__(self):
        self.current_states = {}

    def start_setup(self):
        """
        setup the environment at the start of each episode
        """
        pass

    def start(self, id):
        """
        returns the initial state of the environment, for agent with 'id'
        """
        pass

    def step(self, id, action):
        """
        returns a triplet [state, reward, is_terminal] after agent with 'id' executed 'action'
        """
        pass