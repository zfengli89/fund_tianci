#coding=utf-8
import numpy as np
import pandas as pd
import random

def start():
    mat = []
    cnt = 0
    for i in range(5):
        tmp = []
        for j in range(6):
            cnt = random.random()
            if i==3 :
                tmp.append(None)
            else:
                tmp.append(cnt)
        mat.append(tmp)
    mat = [[15881.007565707221, 15602.570740461997, 15607.291425775036], [15809.223173678092, None, None], [15770.275316561625, None, None]]
    mat = pd.DataFrame(mat)
    print mat
    print mat.stack()
    print mat.stack().idxmin()

if __name__ == '__main__':
    start()

