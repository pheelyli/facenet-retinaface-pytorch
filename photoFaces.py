
class Photos:
    def __init__(self, dirname):
        self.dirname = dirname
        self.photos = []


class Photo:
    def __init__(self, filename):
        self.filename = filename
        self.faces = []


class Face:
    def __init__(self, personName,x,y,w,h):
        self.personName = personName
        self.x = x
        self.y = y
        self.w = w
        self.h = h