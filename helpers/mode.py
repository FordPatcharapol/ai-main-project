from itertools import islice
from statistics import mode
from math import ceil

class Mode:
    def __init__(self, name, list_size):
        self.name = name
        self.list_size = list_size
        self.list = []
        self.mode = None

    def addList(self, text):

        self.list.append(text)
        self.list = list(islice(reversed(self.list), 0, self.list_size))
        self.list.reverse()

        return self.list

    def getMode(self):

        if len(self.list) >=  ceil(self.list_size/2):
            self.mode = mode(self.list)

        return self.mode

    def clearList(self):
        self.list = []
        self.mode = None
