from Monitor import Monitor

class Recorder(Monitor):
    def __init__(self):
        super(Recorder, self).__init__()
        self.records = {}

    def record_event(self, event):
        row = [event.count]
        row.extend(event.values)
        self.records[event.name].append(row)

    def follow(self, event):
        # setup records for this event, add header
        header = ['number']
        header.extend(event.arguments)
        self.records[event.name] = [header]
        handler = lambda: self.record_event(event)
        super(Recorder, self).follow(event, handler)