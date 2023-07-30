from __future__ import division
from random import seed, shuffle
import random
import math
import os
from collections import defaultdict
from sklearn import svm
import os,sys
import urllib2
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
import numpy as np
import random
import time
from scipy.optimize import basinhopping
import config
from sklearn.externals import joblib
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from scipy.linalg import svd


random.seed(time.time())
start_time = time.time()

params = config.params
input_bounds = config.input_bounds

#***********************************************************************************************

df = pd.read_csv('data/Credit.txt')
input_df = df.iloc[:, :-1]
output_df = df.iloc[:,-1:]

dictSize =300

arr = input_df.to_numpy()
allPoints = np.transpose(arr)

#***********************************************************************************************


init_prob = 0.5

direction_probability = [init_prob] * params
direction_probability_change_size = 0.001

param_probability = [1.0/params] * params
param_probability_change_size = 0.001

sensitive_param = config.sensitive_param
name = 'sex'
cov = 0

perturbation_unit = config.perturbation_unit

threshold = config.threshold

global_disc_inputs = set()
global_disc_inputs_list = []

local_disc_inputs = set()
local_disc_inputs_list = []

final_inputs_list = []

tot_inputs = set()

local_iteration_limit = 1000


classifier_name = config.classifier_name

model = joblib.load(classifier_name)

def normalise_probability():
    probability_sum = 0.0
    for prob in param_probability:
        probability_sum = probability_sum + prob

    for i in range(params):
        param_probability[i] = float(param_probability[i])/float(probability_sum)

class Local_Perturbation(object):

    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        param_choice = np.random.choice(xrange(params) , p=param_probability)
        act = [-1, 1]
        direction_choice = np.random.choice(act, p=[direction_probability[param_choice], (1 - direction_probability[param_choice])])

        if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
            direction_choice = np.random.choice(act)

        x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

        x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

        ei = evaluate_input(x)

        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            direction_probability[param_choice] = min(
                direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            direction_probability[param_choice] = max(
                direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

        if ei:
            param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
            normalise_probability()
        else:
            param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size, 0)
            normalise_probability()

        return x

def evaluate_input(inp):
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

    # return (abs(out0 - out1) > threshold)
    # for binary classification, we have found that the
    # following optimization function gives better results
    return abs(out1 + out0)

def evaluate_global(inp):
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

    tot_inputs.add(tuple(map(tuple, inp0)))

    if (abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in global_disc_inputs):
        global_disc_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs_list.append(inp0.tolist()[0])
        fin = inp0.tolist()[0]
        if(out1[0] == 1):
            fin.append(1)
        else:
            fin.append(0)
        final_inputs_list.append(fin)

    # return not abs(out0 - out1) > threshold
    # for binary classification, we have found that the
    # following optimization function gives better results
    return abs(out1 + out0)


def evaluate_local(inp):
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

    tot_inputs.add(tuple(map(tuple, inp0)))

    if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
        and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
        local_disc_inputs.add(tuple(map(tuple, inp0)))
        local_disc_inputs_list.append(inp0.tolist()[0])
        fin = inp0.tolist()[0]
        fin.append(out1[0])
        final_inputs_list.append(fin)

    # return not abs(out0 - out1) > threshold
    # for binary classification, we have found that the
    # following optimization function gives better results
    return abs(out1 + out0)


#initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
minimizer = {"method": "L-BFGS-B"}

local_perturbation = Local_Perturbation()

global_dict = []
local_dict = []
disc_input_dict = []
total_input_dict = []

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

max_it = 3 #for k-svd

for it in range(3):
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

A_tilda =  np.dot(trainDict, X_sparse)
subt = np.subtract(testDict, A_tilda)

global_df = pd.DataFrame(np.transpose(trainDict), columns = list(input_df.columns.values) )
it = 0
for col in global_df.columns:
    global_df = global_df[global_df[col] >= input_bounds[it][0]]
    global_df = global_df[global_df[col] <= input_bounds[it][1]]
    it += 1

for it in xrange(global_df.shape[0]):
    x = global_df.iloc[it].tolist()
    evaluate_global(x)

local_disc_inputs = set()
local_disc_inputs_list = []

tot_inputs = set()

init_prob = 0.5
direction_probability = [init_prob] * params
direction_probability_change_size = 0.001

param_probability = [1.0/params] * params
param_probability_change_size = 0.001

global_dict.append(len(global_disc_inputs_list)/global_df.shape[0]*100)

global_it = 0
global_iteration_limit = global_df.shape[0]

for inp in global_disc_inputs_list:
    basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                niter=local_iteration_limit)

local_dict.append(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs))*100)
disc_input_dict.append(len(global_disc_inputs_list)+len(local_disc_inputs_list))
total_input_dict.append(len(tot_inputs))

print ""
print "Final:"
print "Average global disc - " + str(np.mean(global_dict))
print "Average local disc - " + str(np.mean(local_dict))
print "Average disc input count - " + str(np.mean(disc_input_dict))
print "Average total input count - " + str(np.mean(total_input_dict))


with open('retrain/RSFair_Decision_Tree_Credit.txt', 'w') as file:
    for row in final_inputs_list:
        file.write(','.join([str(item) for item in row]))
        file.write('\n')
