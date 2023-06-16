#Adult Config Start#
""""
params = 13

sensitive_param = 9 # Starts at 1.

input_bounds = []
input_bounds.append([1, 9])
input_bounds.append([0, 7])
input_bounds.append([0, 39])
input_bounds.append([0, 15])
input_bounds.append([0, 6])
input_bounds.append([0, 13])
input_bounds.append([0, 5])
input_bounds.append([0, 4])
input_bounds.append([0, 1])
input_bounds.append([0, 99])
input_bounds.append([0, 39])
input_bounds.append([1, 99])
input_bounds.append([0, 39])

#classifier_name = 'models/Decision_Tree_Census.pkl'
classifier_name = 'models/Random_Forest_Census.pkl'
#classifier_name = 'models/MLP_Census.pkl'

threshold = 0

perturbation_unit = 1

#retraining_inputs = "retrain/Aequitas_Random_Forest_Census.txt"
retraining_inputs = "retrain/RSFair_Decision_Tree_Census.txt"
"""
#Adult Config End#

#Census Config Start#
"""
params = 13

sensitive_param = 9 # Starts at 1.

input_bounds = []
input_bounds.append([1, 9])
input_bounds.append([0, 7])
input_bounds.append([0, 39])
input_bounds.append([0, 15])
input_bounds.append([0, 6])
input_bounds.append([0, 13])
input_bounds.append([0, 5])
input_bounds.append([0, 4])
input_bounds.append([0, 1])
input_bounds.append([0, 99])
input_bounds.append([0, 39])
input_bounds.append([1, 99])
input_bounds.append([0, 39])

#classifier_name = 'models/Decision_Tree_Census.pkl'
classifier_name = 'models/Random_Forest_Census.pkl'
#classifier_name = 'models/MLP_Adult.pkl'

threshold = 0

perturbation_unit = 1
"""
#Census Config End#


#Credit Config Start#
#params = 13
params = 20

sensitive_param = 9 # Starts at 1.
"""
input_bounds = []
input_bounds.append([1, 9])
input_bounds.append([0, 7])
input_bounds.append([0, 39])
input_bounds.append([0, 15])
input_bounds.append([0, 6])
input_bounds.append([0, 13])
input_bounds.append([0, 5])
input_bounds.append([0, 4])
input_bounds.append([0, 1])
input_bounds.append([0, 99])
input_bounds.append([0, 39])
input_bounds.append([1, 99])
input_bounds.append([0, 39])
"""
input_bounds = []
input_bounds.append([0, 3])
input_bounds.append([4, 72])
input_bounds.append([0, 4])
input_bounds.append([0, 10])
input_bounds.append([3, 184])
input_bounds.append([0, 4])
input_bounds.append([0, 4])
input_bounds.append([1, 4])
input_bounds.append([0, 1])
input_bounds.append([0, 2])
input_bounds.append([1, 4])
input_bounds.append([0, 3])
input_bounds.append([19, 75])
input_bounds.append([0, 2])
input_bounds.append([0, 2])
input_bounds.append([1, 4])
input_bounds.append([0, 3])
input_bounds.append([1, 2])
input_bounds.append([0, 1])
input_bounds.append([0, 1])

classifier_name = 'models/Decision_Tree_Credit.pkl'

threshold = 0

perturbation_unit = 1

#retraining_inputs = "retrain/Aequitas_Random_Forest_Census.txt"
retraining_inputs = "retrain/RSFair_Decision_Tree_Credit.txt"

#Credit Config End#
