import random
import numpy as np

class CommonDataGenerator:
    def __init__(self):
        pass

    @staticmethod
    def random_int(min, max):
        return random.randint(min, max)

    @staticmethod
    def random_float(min, max):
        return min + (max-min)*np.random.random() # generates a floating number in [min, max]

    @staticmethod
    def random_ipoint(min, max, dim):
        return np.random.randint(min, max, dim)

    @staticmethod
    def random_fpoint(min, max, dim):
        return min + (max-min)*np.random.random(dim) # generates a floating number in [min, max]

    @staticmethod
    def random_ipoint_list(min, max, dim, size):
        return np.random.randint(min, max, (dim, size))

    @staticmethod
    def random_fpoint_list(min, max, dim, size):
        return min + (max-min)*np.random.random((size, dim)) # generates a floating number list in [min, max]^size
