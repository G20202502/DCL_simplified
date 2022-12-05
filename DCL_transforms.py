import DCL_functional as F

class Randomswap(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.swap(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)