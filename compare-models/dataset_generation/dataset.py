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
def generate_all_bundle_value_pairs(world, k=26144):
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

    bundle_value_pairs = np.array(
        [list(x) + [world.calculate_value(bidder_id, x) for bidder_id in N] for x in tqdm(bundle_space)])
    return (bundle_value_pairs)



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

    #print_all = args.print
    #save = args.save
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
    domain = hasattr(PySats.getInstance(),"create_"+str(args.mode))(seed=1)
    print("FOUND DOMAIN")
    # use the GenericWrapper which uses goods with multiple items per good
    # a bundle is not anymore a binary vector but a vector of integers
    if args.mode == "mrvm" or "srvm":
        domain = GenericWrapper(domain)
    bids = generate_all_bundle_value_pairs(domain, k=num_bids)
    #bids = domain.get_uniform_random_bids(bidder_id,num_bids)
    # pickle dump the bids to a file named by mvrn and bidder_id and num_bids using pickle
    if args.save:
        pickle.dump(bids, open("datasets/"+str(args.mode)+"_"+str(bidder_id)+"_"+str(num_bids)+".pkl", "wb"), -1)
        print("Saved")
        exit()

    if args.mode =='mrvm':
        if save:
            if not os.path.exists("datasets/mrvm"):
                os.makedirs("datasets/mrvm")
        bidder_id = args.bidder_id[0]
        num_bids = args.num_bids[0]
        seed = args.seed
        print("Current mode : MRVN")
        # create an MRVM instance
        mrvm = PySats.getInstance().create_mrvm(seed=1)
        # use the GenericWrapper which uses goods with multiple items per good
        # a bundle is not anymore a binary vector but a vector of integers
        mrvm_generic = GenericWrapper(mrvm)

        bids = mrvm_generic.get_uniform_random_bids(bidder_id,num_bids)

        # pickle dump the bids to a file named by mvrn and bidder_id and num_bids using pickle
        if save:
            pickle.dump(bids, open("datasets/mrvm/mrvm_"+str(bidder_id)+"_"+str(num_bids)+".pkl", "wb"), -1)

    if args.mode == 'srvm':
        if save:
            if not os.path.exists("datasets/srvm"):
                os.makedirs("datasets/srvm")
        bidder_id = args.bidder_id[0]
        num_bids = args.num_bids[0]
        seed = args.seed
        print("Current mode : SRVN")
        # create an MRVM instance
        srvm = PySats.getInstance().create_srvm(seed=1)
        # use the GenericWrapper which uses goods with multiple items per good
        # a bundle is not anymore a binary vector but a vector of integers
        srvm_generic = GenericWrapper(srvm)

        bids = srvm_generic.get_uniform_random_bids(bidder_id, num_bids)

        # pickle dump the bids to a file named by vrn and bidder_id and num_bids using pickle
        if save:
            pickle.dump(bids, open("datasets/srvm/srvm_" + str(bidder_id) + "_" + str(num_bids) + ".pkl", "wb"), -1)

    if args.mode == 'gsvm':
        if save:
            if not os.path.exists("datasets/gsvm"):
                os.makedirs("datasets/gsvm")
        bidder_id = args.bidder_id[0]
        num_bids = args.num_bids[0]
        seed = args.seed
        print("Current mode : GSVM")
        # create an GSVM instance
        gsvm = PySats.getInstance().create_gsvm(seed=1)
        # use the GenericWrapper which uses goods with multiple items per good
        # a bundle is not anymore a binary vector but a vector of integers
        #gsvm_generic = GenericWrapper(gsvm)

        bids = gsvm.get_uniform_random_bids(bidder_id, num_bids)

        # pickle dump the bids to a file named by mvrn and bidder_id and num_bids using pickle
        if save:
            pickle.dump(bids, open("datasets/gsvm/gsvm_" + str(bidder_id) + "_" + str(num_bids) + ".pkl", "wb"),
                        -1)
    if args.mode == 'lsvm':
        if save:
            if not os.path.exists("datasets/lsvm"):
                os.makedirs("datasets/lsvm")
        bidder_id = args.bidder_id[0]
        num_bids = args.num_bids[0]
        seed = args.seed
        print("Current mode : LSVM")
        # create an LSVM instance
        lsvm = PySats.getInstance().create_lsvm(seed=1)
        # use the GenericWrapper which uses goods with multiple items per good
        # a bundle is not anymore a binary vector but a vector of integers
        # lsvm_generic = GenericWrapper(lsvm)

        bids = lsvm.get_uniform_random_bids(bidder_id, num_bids)

        # pickle dump the bids to a file named by mvrn and bidder_id and num_bids using pickle
        if save:
            pickle.dump(bids, open("datasets/lsvm/lsvm_" + str(bidder_id) + "_" + str(num_bids) + ".pkl", "wb"),
                    -1)
            '''
            if print_all:
                # Number of goods in original pySats: 98
                print(len(mrvm.get_good_ids()))
                #Number of goods in GenericWrapper: 42
                print(len(mrvm_generic.get_good_ids()))

                # the GenericWrapper has additional attributes that allow to map goods to licences (single items):
                # i.e. licence with id 2 (pysats) maps to good with id 7 (generic wrapper)
                print(mrvm_generic.licence_to_good[2])  # maps 98 licenses to 42 goods
                # i.e the good with id 7 (generic wrapper) maps to licences 2 and 16 (pysats)
                print(mrvm_generic.good_to_licence[7])

                if has_cyplex:
                    # keys: goods, values: however many goods map to it
                    capacities = {i: len(mrvm_generic.good_to_licence[i]) for i in range(len(mrvm_generic.good_to_licence))}
                    capacities2 = mrvm_generic.get_capacities()

                    #currently no cyplex
                    # compare the efficient allocation
                    mrvm_eff = mrvm.get_efficient_allocation()
                    mrvm_generic_eff = mrvm_generic.get_efficient_allocation()
                    # efficient values are the same
                    print(mrvm_eff[1])
                    print(mrvm_generic_eff)


                    # bidder allocation of original pysats (good refers to a licence)
                    mrvm_eff[0][1]
                    # bidder allocation of generic wrapper (good refers to generic good), allocation contains additional good_count
                    # (i.e. number of goods allocated in the same order as good_ids)
                    mrvm_generic_eff[0][1]

                    # perform a demand query
                    price = np.zeros(len(mrvm_generic.get_good_ids()))
                    demand = mrvm_generic.get_best_bundles(1,price,2,allow_negative=True)
                    print(demand[0])
                    print(len(demand[0]))

            # random queries work as expected
            bid = mrvm_generic.get_uniform_random_bids(1,1)[0]
            value = mrvm_generic.calculate_value(1, demand[0])

    # ensuring that the value in 2 represetations is the same
    bundle_extended_representation = [0 for i in range(len(mrvm.get_good_ids()))]
    for i in range(len(demand[0])):
        license_mapping = mrvm_generic.good_to_licence[i]
        items_requested = demand[0][i]
        for j in range(items_requested):
            bundle_extended_representation[license_mapping[j]] = 1


    value_extended_representation = mrvm.calculate_value(1, bundle_extended_representation)

    print('Difference between the values in the 2 representations:', value - value_extended_representation)

    print('---Starting SRVM test ---')

    srvm = PySats.getInstance().create_srvm(1)
    srvm_generic = GenericWrapper(srvm)

    print('Length of good ids without generic wrapper:', len(srvm.get_good_ids()))
    #Number of goods in GenericWrapper: 42
    print('Lnegth of good ids WITH generic wrapper:',len(srvm_generic.get_good_ids()))


    # keys: goods, values: however many goods map to it
    capacities = {i: len(srvm_generic.good_to_licence[i]) for i in range(len(srvm_generic.good_to_licence))}
    capacities2 = srvm_generic.get_capacities()

    print('Generic capacities:', capacities)
    set_trace()

    # compare the efficient allocation
    srvm_eff = srvm.get_efficient_allocation()
    srvm_generic_eff = srvm_generic.get_efficient_allocation()



    # efficient values are the same
    print(srvm_eff[1])
    print(srvm_generic_eff)


    # bidder allocation of original pysats (good refers to a licence)
    srvm_eff[0][1]
    # bidder allocation of generic wrapper (good refers to generic good), allocation contains additional good_count
    # (i.e. number of goods allocated in the same order as good_ids)
    srvm_generic_eff[0][1]

    # perform a demand query
    price = np.zeros(len(srvm_generic.get_good_ids()))
    demand = srvm_generic.get_best_bundles(1,price,2,allow_negative=True)
    print(demand[0])
    print(len(demand[0]))

    # random queries work as expected
    bid = srvm_generic.get_uniform_random_bids(1,1)[0]


    value = srvm_generic.calculate_value(1, demand[0])


    # ensuring that the value in 2 represetations is the same
    bundle_extended_representation = [0 for i in range(len(srvm.get_good_ids()))]
    for i in range(len(demand[0])):
        license_mapping = srvm_generic.good_to_licence[i]
        items_requested = demand[0][i]
        for j in range(items_requested):
            bundle_extended_representation[license_mapping[j]] = 1


    value_extended_representation = srvm.calculate_value(1, bundle_extended_representation)

    print('Difference between the values in the 2 representations:', value - value_extended_representation
    '''
if __name__ == "__main__":
    has_cyplex = False
    print("Flag has_cyplex is set to "+ str(has_cyplex))
    main()
