# I/O functionality for Bayesian Retrement code

from Annotations import Annotations, Annotation

class Receiver():

    def __init__(self):
        pass

    def annotations(self):
        # Receive new annotations and return a new Annotations list
        # NOTE: Annotations implicity contain classifiers

class Transmitter():
    pass

class CaesarReceiver(Receiver):
    pass

class CaesarTransmitter(Transmitter):
    pass

class Storage():
    pass

class SQLiteStorage(Storage):
    pass

class FileStorage(Storage):
    pass
