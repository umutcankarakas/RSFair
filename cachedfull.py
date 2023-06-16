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

random.seed(time.time())
start_time = time.time()

#***********************************************************************************************

df = pd.read_csv('data/cleaned_train')
input_df = df.iloc[:, :-1]
output_df = df.iloc[:,-1:]

def interval_checker(value, borders):
  for i in range(len(borders)):
    if value > borders[i]:
        continue
    else: 
        return pd.Interval(left=borders[i-1], right=borders[i])

  return pd.Interval(left=borders[len(borders)-2], right=borders[len(borders)-1])

def weight_checker(value, borders, weights):
  interval = interval_checker(value, borders)
  return weights[interval]

def value_calculator(record, bin_limits, weight_bins):
  total = 0
  it = 0
  for col in list(input_df.columns.values):
    total += weight_checker(record[it], bin_limits[col], weight_bins[col])
    it+=1

  return total

weight_bins = {}
bins = {}
bin_limits = {}
edges = {}

max_value = input_df.shape[1]

for col in input_df.columns:
  bins, bin_limits[col] = pd.cut(input_df[col], bins=4, retbins = True)
  for i in range(len(bin_limits[col])):
    bin_limits[col][i] = round(bin_limits[col][i], 3)
  weight_bins[col] = pd.Series(bins).value_counts(normalize=True).sort_index(ascending=True)

#***********************************************************************************************

input_bounds = config.input_bounds
classifier_name = config.classifier_name

model = joblib.load(classifier_name)

init_prob = 0.5
params = config.params
direction_probability = model.feature_importances_
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

tot_inputs = set()

global_iteration_limit = 2000
local_iteration_limit = 1000

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


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize

        try_count = 0
        max_x = []
        max_x_value = 0
        while(try_count < 100):
            try_count += 1
            for i in xrange(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = 0

            if(value_calculator(x, bin_limits, weight_bins)/max_value*100 > 60):
               max_x = x
               break
            
            if(max_x_value < value_calculator(x, bin_limits, weight_bins)/max_value*100):
                max_x_value = value_calculator(x, bin_limits, weight_bins)/max_value*100
                max_x = x

        # print x
        return max_x


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

    # return not abs(out0 - out1) > threshold
    # for binary classification, we have found that the
    # following optimization function gives better results
    return abs(out1 + out0)


initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
minimizer = {"method": "L-BFGS-B"}

global_discovery = Global_Discovery()
local_perturbation = Local_Perturbation()

global_dict = {}
local_dict = {}
disc_input_dict = {}
total_input_dict = {}

for i in xrange(100):
    print i
    global_disc_inputs = set()
    global_disc_inputs_list = []

    local_disc_inputs = set()
    local_disc_inputs_list = []

    tot_inputs = set()

    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0/params] * params
    param_probability_change_size = 0.001

    basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
             niter=global_iteration_limit)


    global_dict[i] = float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs))*100

    for inp in global_disc_inputs_list:
        basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                 niter=local_iteration_limit)

    local_dict[i] = float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs))*100
    disc_input_dict[i] = len(global_disc_inputs_list)+len(local_disc_inputs_list)
    total_input_dict[i] = len(tot_inputs)

dicts = global_dict, local_dict, disc_input_dict, total_input_dict

tot_global_perc = sum(global_dict.values())
tot_local_perc = sum(local_dict.values())
tot_disc_inp = sum(disc_input_dict.values())
tot_inp = sum(total_input_dict.values())

print "Average global disc - " + str(tot_global_perc/100)
print "Average local disc - " + str(tot_local_perc/100)
print "Average disc input count - " + str(tot_disc_inp/100)
print "Average total input count - " + str(tot_inp/100)