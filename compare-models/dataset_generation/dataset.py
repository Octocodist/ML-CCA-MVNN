import sys

from pysats import PySats
#from sympy.polys.polyoptions import Gen
#from pdb import set_trace
import os

# make sure you set the classpath before loading any other modules
PySats.getInstance()

import numpy as np
import pickle
from pysats_ext import GenericWrapper
import argparse
import itertools
import time
import random

# libs--
import numpy as np
from tqdm import tqdm

# order 0 if model_name not in ['SRVM', 'MRVM'] else 2)
def generate_all_bundle_value_pairs(world, k=4):
    print("STARTED SAMPLING")
    N = world.get_bidder_ids()
    M = world.get_good_ids()

    if world == 'srvm' or world == 'mrvm':
        bundle_space = [np.random.choice([0, 1], len(M)) for _ in range(k)]
        # Only use unique samples.
        bundle_space = np.unique(np.array(bundle_space), axis=0)
    else:
        #this creates a list of lists of length len(M) with all possible combinations of 0 and 1
        #this calculates the cartesian product of the list [0,1] with itself len(M) times
        bundle_space = list(itertools.product([0, 1], repeat=len(M)))

    #bundle_value_pairs = np.array(
    #    [list(x) + [world.calculate_value(bidder_id, x) for bidder_id in N] for x in tqdm(bundle_space)])
    values = [[world.calculate_value(bidder_id, x) for bidder_id in N] for x in tqdm(bundle_space)]
    return bundle_space, values


def init_parser():
    parser = argparse.ArgumentParser(description='Generate datasets for the ML-CCA-MVNN project')
    parser.add_argument('-m','--mode',  type=str, help='Choose mode to sample from mrvm, srvm', choices=['mrvm','srvm','gsvm','lsvm'])
    parser.add_argument('-b','--bidder_id',nargs=1, type=int, help='Define bidder id')
    parser.add_argument('-n','--num_bids',nargs=1 ,type=int, help='Define bidder id')
    parser.add_argument('-sd','--seed',nargs=1 ,type=int, help='Define seed', default=1 )
    parser.add_argument('-s','--save',action='store_true', help='Save the generated dataset', default=False)
    parser.add_argument('-p','--print', action= 'store_true', help='Print the generated dataset', default=False)
    parser.add_argument('-num', '--number_of_instances', type=int, default=1, help='Num. training data  (1)')

    return parser
def main():
    print("--Start Program--")
    parser = init_parser()
    args = parser.parse_args()
    print(args)


    print("Print all is set to " + str(args.print))
    print("Save is set to " + str(args.save))
    print("Arguments passed: " + str(args))
    if args.save:
        #check if folder exists and create it if not
        if not os.path.exists("datasets"):
            os.makedirs("datasets")

    if args.save:
        if not os.path.exists("datasets/"+str(args.mode)):
            os.makedirs("datasets/"+str(args.mode))
    bidder_id = args.bidder_id[0]
    num_bids = args.num_bids[0]
    seed = args.seed
    print("Current mode :"+ str(args.mode))
    # create an instance
    domain = getattr(PySats.getInstance(),"create_"+str(args.mode))(seed=1)
    print("FOUND DOMAIN")
    # use the GenericWrapper which uses goods with multiple items per good
    # a bundle is not anymore a binary vector but a vector of integers
    if args.mode == "mrvm" or args.mode == "srvm":
        domain = GenericWrapper(domain)
    bundles, values = generate_all_bundle_value_pairs(domain, k=num_bids)
    #bids = domain.get_uniform_random_bids(bidder_id,num_bids)
    # pickle dump the bids to a file named by mvrn and bidder_id and num_bids using pickle
    if args.save:
        pickle.dump((bundles,values), open("datasets/"+str(args.mode)+"/"+str(args.mode)+"_"+str(bidder_id)+"_"+str(num_bids)+".pkl", "wb"), -1)
        print("Saved")

if __name__ == "__main__":
    has_cyplex = False
    print("Flag has_cyplex is set to "+ str(has_cyplex))
    main()
