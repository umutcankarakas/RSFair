import config
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals import joblib

sensitive_param = config.sensitive_param

X = []
Y = []
i = 0
sensitive = {}
sens = []
with open("data/Adult.txt", "r") as ins:
    for line in ins:
        line = line.strip()
        line1 = line.split(',')
        if (i == 0):
            i += 1
            continue
        L = map(int, line1[:-1])
        sens.append(L[sensitive_param - 1])
        # L[sens_arg-1]=-1
        X.append(L)

        if (int(line1[-1]) == 0):
            Y.append(-1)
        else:
            Y.append(1)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test


#Decision Tree Part *********************************************


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Save the model as a pickle in a file
joblib.dump(clf, 'models/Decision_Tree_Adult.pkl')


#Random Forest Part **************************************************

# Create Random Forest classifer object
rf = RandomForestClassifier()

# Train Random Forest Classifer
rf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = rf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Save the model as a pickle in a file
joblib.dump(rf, 'models/Random_Forest_Adult.pkl')

#MLP Part **************************************************

# Create MLP classifer object
mlp = MLPClassifier()

# Train MLP Classifer
mlp.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = mlp.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Save the model as a pickle in a file
joblib.dump(mlp, 'models/MLP_Adult.pkl')

