class Monitor(object):
    def __init__(self):
        self.handlers = {}
        self.active = False

    def follow(self, event, handler):
        event.followers.append(self)
        self.handlers[event.name] = handler

    def handle(self, event):
        if self.active:
            self.handlers[event.name]()

    def activate(self, active=True):
        self.active = active