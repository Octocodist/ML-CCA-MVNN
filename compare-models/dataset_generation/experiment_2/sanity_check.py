import pickle 
import pandas as pd
import glob
from copy import deepcopy




#nbids=25000
#nbids=100
#seed=103
#dataset="mrvm"
filepath = "/cluster/home/filles/masterthesis/ML-CCA-MVNN/compare-models/dataset_generation/experiment_2/blogfeedback/"

train = pd.read_csv(filepath + "blogData_train.csv", header=None)
 # 0-280 features and 1 is result
 # A51 - A54, A56 - A59 are monotonic 
 
#path =r'../data/test/raw' 

train_mono = train.iloc[:,51:60].copy(deep=True)
train_mono = train_mono.drop(columns=[55], axis = 1)

y_train = train.iloc[:,-1].copy(deep=True)

train = train.drop(columns = [51,52,53,54,56,57,58,59,180], axis = 1)

# save pickle 
with open(filepath+"blog_train.pkl","wb") as file:
    pickle.dump([train, train_mono, y_train],file)


allFiles = glob.glob(filepath + "test/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test._append(pd.read_csv(file_, header=None))


test_mono = test.iloc[:,51:60].copy(deep=True)
test_mono = test_mono.drop(columns=[55], axis = 1)

y_test = test.iloc[:,-1].copy(deep=True)

test = test.drop(columns = [51,52,53,54,56,57,58,59,180], axis = 1)

# save pickle 
with open(filepath+"blog_test.pkl","wb") as file:
    pickle.dump([test, test_mono, y_test],file)
print("Done")     
