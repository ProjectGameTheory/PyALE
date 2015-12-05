from __future__ import print_function
from Monitor import Monitor

class Printer(Monitor):
    def print_event(self, event):
        to_print = str(event.count) + ': \t'
        for i in range(len(event.arguments)):
            to_print += event.arguments[i] + ': ' + str(event.values[i]) + '\t'
        print(to_print)

    def follow(self, event):
        handler = lambda: self.print_event(event)
        super(Printer, self).follow(event, handler)