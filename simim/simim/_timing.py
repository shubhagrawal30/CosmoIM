import time

import numpy as np

class _timer():
    def __init__(self,longtime=np.inf,longlongtime=np.inf):
        self.t0 = 0.0
        self.t1 = 0.0
        self.longtime = longtime
        self.longlongtime = longlongtime

    def start(self,item=''):
        self.item = item
        self.t0 = time.perf_counter()
    def lap(self,say=True):
        self.t1 = time.perf_counter()
        t = self.t1-self.t0

        if say:
            text = "\033[1m"+"Timer:"+"\033[0m"+" {} took {}s".format(self.item,t)
            if t > self.longlongtime:
                text = '\033[91m'+text
            elif t > self.longtime:
                text = '\033[93m'+text
            print(text)
        return t
