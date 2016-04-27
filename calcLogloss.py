#!/usr/bin/python
import sys
import math

# argv: 1 - label file, 2 - prediction file

labels = map(lambda x: x.strip().split(','), open(sys.argv[1]))
lbmapping = {}
for l in labels:
    lbmapping[l[0]] = int(l[1])

preds = map(lambda x: x.strip().split(','), open(sys.argv[2]))
ll = 0.0
for l in preds:
    tmpp = float(l[lbmapping[l[0]]+1])
    if tmpp == 0.0:
        tmpp = 1e-15
    ll += math.log(tmpp)
ll /= 50000.0

print ll
