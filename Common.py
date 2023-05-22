import math


def positiveMod(a, b):
    ret = math.fmod(a, b)

    if ret < 0:
        ret += b

    return ret