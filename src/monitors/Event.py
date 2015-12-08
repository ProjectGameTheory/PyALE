
class Event(object):
    def __init__(self, name, arguments=None):
        self.followers = []
        self.name = name
        self.values = None
        self.arguments = arguments
        self.count = 0

    def notify(self, *args):
        self.count += 1
        self.values = args
        for follower in self.followers:
            follower.handle(self)

episode_ended = Event('episode', arguments=['id', 'steps', 'reward'])