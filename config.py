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
input_bounds.append([0, 99])
input_bounds.append([0, 39])

classifier_name = 'models/Decision_Tree_Cleaned_Train.pkl'

threshold = 0

perturbation_unit = 1

retraining_inputs = "Retrain_Example_File.txt"