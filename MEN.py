#Evaluate semantic space against MEN dataset

import sys
import utils
from scipy import stats
import numpy as np
from math import sqrt


#Note: this is scipy's spearman, without tie adjustment
def spearman(x,y):
	return stats.spearmanr(x, y)[0]

def readMEN(annotation_file):
  pairs=[]
  humans=[]
  f=open(annotation_file,'r')
  for l in f:
    l=l.rstrip('\n')
    items=l.split()
    pairs.append((items[0],items[1]))
    humans.append(float(items[2]))
  f.close()
  return pairs, humans


def compute_men_spearman(dm_dict, annotation_file):
    pairs, humans=readMEN(annotation_file)
    system_actual=[]
    human_actual=[]
    count=0
    for i in range(len(pairs)):
        human=humans[i]
        a,b=pairs[i]
        if a in dm_dict and b in dm_dict:
            cos=utils.cosine_similarity(dm_dict[a],dm_dict[b])
            system_actual.append(cos)
            human_actual.append(human)
            count+=1
    sp = spearman(human_actual,system_actual)
    return sp,count

