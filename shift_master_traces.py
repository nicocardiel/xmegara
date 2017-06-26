from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='shift_master_traces')
    # positional parameters
    parser.add_argument("initial_traces",
                        help="Initial JSON file with master traces",
                        type=argparse.FileType('r'))
    parser.add_argument("final_traces",
                        help="Final JSON file with master traces",
                        type=argparse.FileType('w'))
    parser.add_argument("yoffset",
                        help="Vertical offset (+upwards, -downwards)",
                        type=float)

    args = parser.parse_args(args=args)

    # read initial traces from JSON file
    bigdict = json.loads(open(args.initial_traces.name).read())

    # introduce vertical offset in all the fibers
    for fiberdict in bigdict['contents']:
        if len(fiberdict['fitparms']) > 0:
            fiberdict['fitparms'][0] += args.yoffset

    # save final traces to JSON file
    with open(args.final_traces.name, 'w') as outfile:
        json.dump(bigdict, outfile, indent=4)

if __name__ == "__main__":

    main()
