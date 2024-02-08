from backbone.my_pipeline import my_pipeline


class DreyeveValidation(object):
    def __init__(self):
        pass
    
    def load_file(self, path:str):
        retval = []
        with open(path, 'r') as file:
            file.readline()
            for line in file.readlines():
                vals = line.split(' ')
                X,Y = float(vals[2]),float(vals[3])
                retval.append((X,Y))
        return retval
    
    def load_files(self, paths):
        retval = []
        for path in paths:
            retval += self.load_file(path)
        return retval




