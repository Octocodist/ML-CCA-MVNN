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

def init_parser():
    parser = argparse.ArgumentParser(description='Generate datasets for the ML-CCA-MVNN project')
    parser.add_argument('-mrvm', nargs=2, type=int,  metavar=('bidder_id', 'num_bids'), help='Generate a dataset for the MRVM with bidder_id and num_bids')
    parser.add_argument('-srvm', nargs=3, metavar=('bidder_id', 'num_bids', 'seed'), help='Generate a dataset for the SRVM with the given parameters')
    parser.add_argument('-s','--save',action='store_true', help='Save the generated dataset', default=False)
    parser.add_argument('-p','--print', action= 'store_true', help='Print the generated dataset', default=False)
    return parser



def main():
    parser = init_parser()
    args = parser.parse_args()
    print(args)

    print_all = args.print
    save = args.save
    print("Print all is set to "+ str(print_all))
    print("Save is set to "+ str(save))
    print("Arguments passed: "+ str(args))
    print("Arguments passed: "+ str(args.mrvm))
    if save:
        #check if folder exists and create it if not
        if not os.path.exists("datasets"):
            os.makedirs("datasets")

    if args.mrvm:
        if save:
            if not os.path.exists("datasets/mrvm"):
                os.makedirs("datasets/mrvm")
        bidder_id = args.bidder_id
        num_bids = args.num_bids
        seed = args.seed
        print("Current mode : MRVN")
        # create an MRVM instance
        mrvm = PySats.getInstance().create_mrvm(seed=1)
        # use the GenericWrapper which uses goods with multiple items per good
        # a bundle is not anymore a binary vector but a vector of integers
        mrvm_generic = GenericWrapper(mrvm)

        bids = mrvm_generic.get_uniform_random_bids(bidder_id=1,num_bids=1)

        # pickle dump the bids to a file named by mvrn and bidder_id and num_bids using pickle
        if save:
            pickle.dump(bids, open("datasets/mrvm/mrvm_"+str(bidder_id)+"_"+str(num_bids)+".pkl", "wb"), -1)
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