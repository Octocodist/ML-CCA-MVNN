import pickle
import numpy as np
import compas_loader as cp
import pandas as pd
from copy import deepcopy 



train, y_train, test, y_test, start_index, cat_length = cp.load_data(get_categorical_info=True)

n = train.shape
print(" num featuers is : " , n )

train_mono = deepcopy(train[:,:4])
train_non_mono = deepcopy(train[:,4:])

print(train_mono.shape)
print(train_non_mono.shape)

n = test.shape
print(" num featuers is : " , n )

test_mono = deepcopy(test[:,:4])
test_non_mono = deepcopy(test[:,4:])

print(test_mono.shape)
print(test_non_mono.shape)


filepath = "/cluster/home/filles/masterthesis/ML-CCA-MVNN/compare-models/dataset_generation/experiment_2/compas"
with open(filepath+"/../compas_train.pkl","wb") as file:
    pickle.dump([pd.DataFrame(train_non_mono), pd.DataFrame(train_mono),pd.DataFrame(y_train)],file)


with open(filepath+"/../compas_test.pkl","wb") as file:
    pickle.dump([pd.DataFrame(test_non_mono), pd.DataFrame(test_mono),pd.DataFrame(y_test)],file)
print("Done")     
