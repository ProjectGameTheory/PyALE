from Recorder import Recorder
import csv

class Writer(Recorder):
    def __init__(self, folderpath = '.'):
        super(Writer, self).__init__()
        self.folderpath = folderpath

    def save(self):
        if self.active:
            for name, records in self.records.items():
                filename = self.folderpath + '/' + name + '.csv'
                with open(filename, 'wb') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(records[0])
                    writer.writerows(records[1:])