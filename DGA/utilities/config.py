from configparser import ConfigParser

class Config:
    __file='config_file.ini'
    __config=None

    def __init__(self):
        self.__config = ConfigParser()
        self.__config.read(self.__file)

    def getModelPath()->str:
        return 