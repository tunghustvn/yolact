'''
This file will optimize the aspect ratio of the anchor boxes
'''

import os
import json
import numpy as np
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Find anchor box')
    parser.add_argument(
        '--annotations', help='Link to annotations file', default=None)
    parser.add_argument(
        '--initmuy', help='The initial muy (3 numbers)',type=float, metavar='N', nargs='+', default=[1.0, 0.5, 0.1])
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def anchor_box(annotations, initmuy):
    json_ann = json.load(open(annotations))

    anno = json_ann['annotations']
    bboxes = []
    for i in anno:
        bboxes.append(i['bbox'])
    np_bboxes = np.array(bboxes, dtype=float)
    asp = np.zeros(len(bboxes))
    asp = np.divide(np_bboxes[:,2],np_bboxes[:,3])
    print(asp)
    
    c = np.zeros(len(asp))    

    muy = initmuy

    last_cluster=c

    while True:
        
        v = np.zeros((len(muy),len(asp)))
        for i in range(len(muy)):
            v[i,:] = (asp - muy[i])**2
        # print(v.mean())
        c = v.argmin(axis=0)

        if (last_cluster == c).all(): 
            break
        last_cluster = c 
        
        err=0
        for k in range(len(muy)):
            s = []
            
            for i in range(len(c)):
                if c[i] == k:
                    s.append(asp[i])
                    err = err + v[k,i]
            print(len(s))
            muy[k] = np.average(s)
            print(muy)
        print(err)


if __name__ == '__main__':

    args = parse_args()
    anchor_box(args.annotations, args.initmuy)
