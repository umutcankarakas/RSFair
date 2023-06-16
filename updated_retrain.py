from sklearn.externals import joblib
import config
import time
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.linalg import svd

def extract_inputs(filename):
    X = []
    Y = []
    i = 0
    neg_count = 0
    pos_count = 0
    with open(filename, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            L = map(int, line1[:-1])
            # L[sens_arg-1]=-1
            X.append(L)

            if (int(line1[-1]) == 0):
                Y.append(-1)
                neg_count = neg_count + 1
            else:
                Y.append(1)
                pos_count = pos_count + 1

    return X, Y

def retrain(X_original, Y_original, X_additional, Y_additional):

    X = np.concatenate((X_original, X_additional), axis = 0)
    Y = np.concatenate((Y_original, Y_additional), axis = 0)

    current_model.fit(X, Y)
    return current_model

def evaluate_input(inp, model):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    inp0 = np.asarray(inp0)
    inp0 = np.reshape(inp0, (1, -1))

    inp1 = np.asarray(inp1)
    inp1 = np.reshape(inp1, (1, -1))

    out0 = model.predict(inp0)
    out1 = model.predict(inp1)

    return (abs(out0 + out1) == 0)

def get_estimate(model, global_count):
    disc_count = 0
    for it in xrange(global_df.shape[0]):
        x = global_df.iloc[it].tolist()
        if evaluate_input(x, model):
            disc_count += 1

    succ_rate = float(disc_count)/global_count*100
    return succ_rate


def retrain_search():
    global current_estimate
    global current_model
    X_retrain, Y_retrain = extract_inputs(retraining_inputs)
    retrain_len = len(X_retrain)
    flag = True

    min_perc = 9999.0
    for i in xrange(7):
        X_additional = []
        Y_additional = []
        retraining_input_set = set()
        additive_percentage = random.uniform(pow(2, i), pow(2, i + 1))
        num_inputs_for_retrain = int((additive_percentage * len(X))/100)

        if (num_inputs_for_retrain > retrain_len and flag):
            num_inputs_for_retrain = retrain_len 
            flag = False
        elif (num_inputs_for_retrain > retrain_len and not flag):
            break

        while (len(retraining_input_set) < num_inputs_for_retrain):
            retraining_input_set.add(random.randint(0, retrain_len - 1))

        for i in retraining_input_set:
            X_additional.append(X_retrain[i])
            Y_additional.append(Y_retrain[i])
        retrained_model = retrain(X_original, Y_original, np.array(X_additional), np.array(Y_additional))
        retrained_estimate = get_estimate(retrained_model, global_df.shape[0])
        if min_perc > retrained_estimate:
            min_perc = retrained_estimate
        """if (retrained_estimate > current_estimate):
            return current_model
        else:
            current_model = retrained_model
            current_estimate = retrained_estimate
            del retrained_estimate
            del retrained_model"""
    
    return min_perc

X, Y = extract_inputs("data/Census.txt")
X_original = np.array(X)
Y_original = np.array(Y)

num_trials = 100
samples = 100

classifier_name = config.classifier_name
current_model = joblib.load(classifier_name)
input_bounds = config.input_bounds
params = config.params
sensitive_param = config.sensitive_param

retraining_inputs = config.retraining_inputs

#****************************************************************
df = pd.read_csv('data/Census.txt')
input_df = df.iloc[:, :-1]
output_df = df.iloc[:,-1:]

dictSize = 300

before = []
after = []

for iteration in range(400):
    print iteration

    arr = input_df.to_numpy()
    allPoints = np.transpose(arr)

    global_disc_inputs = set()
    global_disc_inputs_list = []
    global_df = pd.DataFrame()

    trainDict = np.zeros(shape=(allPoints.shape[0], dictSize))

    test_idxs = [i for i in range(allPoints.shape[1])]
    train_idxs = [test_idxs.pop(random.randrange(len(test_idxs))) for _ in range(dictSize)]

    testDict = np.zeros(shape=(allPoints.shape[0], len(test_idxs)))
    for layer in range(allPoints.shape[0]):
        for idx in range(len(test_idxs)):
            testDict[layer][idx] = allPoints[layer, test_idxs[idx]]

    trainDict = np.zeros(shape=(allPoints.shape[0], len(train_idxs)))
    for layer in range(allPoints.shape[0]):
        for idx in range(len(train_idxs)):
            trainDict[layer][idx] = allPoints[layer, train_idxs[idx]]

    max_coefs = 1 #atoms 1 to 10
    max_it = 3 #for k-svd


    for it in range(max_it):
        # Sparse coding step using OMP algorithm
        X_sparse = np.zeros(shape=(trainDict.shape[1], testDict.shape[1]))
        A_tilda = np.zeros(shape=testDict.shape) #Recreated hyperspectral image
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=1, normalize=False) 

        for i in range(testDict.shape[1]):
            omp.fit(trainDict, testDict[:,i:i+1])
            coef = omp.coef_
            X_sparse[:, i] = coef
            (idx_r,) = coef.nonzero()

        #K-SVD Start
        for j in range(dictSize):
            # Find nonzero elements in column j of X_sparse
            I = np.nonzero(X_sparse[j, :])[0]
                    
            if len(I) == 0:
                continue
                    
            # Update dictionary column j
            E = testDict[:, I] - np.dot(trainDict, X_sparse[:, I]) + np.outer(trainDict[:, j], X_sparse[j, I])
            U, s, Vt = svd(E)
            trainDict[:, j] = U[:, 0]
            X_sparse[j, I] = s[0] * Vt[0, :]
            #K-SVD End

    global_df = pd.DataFrame(np.transpose(trainDict), columns = list(input_df.columns.values) )
    it = 0
    for col in global_df.columns:
        global_df = global_df[global_df[col] >= input_bounds[it][0]]
        global_df = global_df[global_df[col] <= input_bounds[it][1]]
        it += 1

#****************************************************************

    print("Current discriminatory percentage:")
    current_estimate = get_estimate(current_model, global_df.shape[0])
    before.append(current_estimate)
    print(np.mean(before))

    print("Retrained discriminatory percentage:")
    retrain_estimate = retrain_search()
    after.append(retrain_estimate)
    print(np.mean(after))

    
    