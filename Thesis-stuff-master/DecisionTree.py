# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:03:11 2019

@author: Robot Hands

This is exerything pertaining to the XGBoost model I used for fault detection on an industrial machine
"""

import warnings

#Turns off deprecation warnings associated with label.py
warnings.filterwarnings("ignore", category = DeprecationWarning)

from pandas import read_csv
import numpy
from numpy import sort

from sklearn.feature_selection import SelectFromModel as SFM

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier as xgb
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import pickle

#load data into arrays
file = 'allData_16-24_05_2017.csv'

df = read_csv(file)
array = df.values

S1_X = array[:, 0: 5]
S1_Y = array[:, 5]
   
S2_X = array[:, 6: 11]
S2_Y = array[:, 11]
   
S3_X = array[:, 12: 17]
S3_Y = array[:, 17]
    
S4_X = array[:, 18: 23]
S4_Y = array[:, 23]
      
S1_X_S2_X = numpy.append(S1_X, S2_X, axis = 1)
S3_X_S4_X = numpy.append(S3_X, S4_X, axis = 1)
    
Sys_X = numpy.append(S1_X_S2_X, S3_X_S4_X, axis = 1)
Sys_Y = array[:, 23].astype('int')


#Split data into training, testing, and validation sets. Give seed for reproductibility, size is the test set size.
seed = 0
size = .30

S1_X_model, S1_X_test, S1_Y_model, S1_Y_test = train_test_split(S1_X, S1_Y, test_size = size, random_state = seed)
S2_X_model, S2_X_test, S2_Y_model, S2_Y_test = train_test_split(S2_X, S2_Y, test_size = size, random_state = seed)
S3_X_model, S3_X_test, S3_Y_model, S3_Y_test = train_test_split(S3_X, S3_Y, test_size = size, random_state = seed)
S4_X_model, S4_X_test, S4_Y_model, S4_Y_test = train_test_split(S4_X, S4_Y, test_size = size, random_state = seed)
Sys_X_model, Sys_X_test, Sys_Y_model, Sys_Y_test = train_test_split(Sys_X, Sys_Y, test_size = size, random_state = seed)

S1_X_train, S1_X_valid, S1_Y_train, S1_Y_valid = train_test_split(S1_X_model, S1_Y_model, test_size = size, random_state = seed)
S2_X_train, S2_X_valid, S2_Y_train, S2_Y_valid = train_test_split(S2_X_model, S2_Y_model, test_size = size, random_state = seed)
S3_X_train, S3_X_valid, S3_Y_train, S3_Y_valid = train_test_split(S3_X_model, S3_Y_model, test_size = size, random_state = seed)
S4_X_train, S4_X_valid, S4_Y_train, S4_Y_valid = train_test_split(S4_X_model, S4_Y_model, test_size = size, random_state = seed)
Sys_X_train, Sys_X_valid, Sys_Y_train, Sys_Y_valid = train_test_split(Sys_X_model, Sys_Y_model, test_size = size, random_state = seed)


#Use XGBoost to show feature importance per station
model1 = xgb().fit(S1_X_train, S1_Y_train)
model2 = xgb().fit(S2_X_train, S2_Y_train)
model3 = xgb().fit(S3_X_train, S3_Y_train)
model4 = xgb().fit(S4_X_train, S4_Y_train)

#Shows the XGBoost-derived feature importances in graph form.
plot_importance(model1)
plot_importance(model2)
plot_importance(model3)
plot_importance(model4)
pyplot.show()

#use sort to find the thresholds for SelectFromModel 
thresholdS1 = sort(model1.feature_importances_)
thresholdS2 = sort(model2.feature_importances_)
thresholdS3 = sort(model3.feature_importances_)
thresholdS4 = sort(model4.feature_importances_)

print("-----------------------------------------------------------------------------")

#Setting up the SelectFromModel
selection1 = SFM(model1, threshold = 0, prefit = True)

#cuts out the unnecessary features from the dataset
select_X1_train = selection1.transform(S1_X_train)

#sets the model and tuning parameters to be used on the feature selected dataset    
selection_model1 = xgb(n_estimators = 40)

#makes a new testset sans the unnecessary features
select_X1_valid = selection1.transform(S1_X_valid)
transform_X1_test = selection1.transform(S1_X_test)

#sets up the evaluation sets to gauge model performance
eval_set1 = [(select_X1_train, S1_Y_train), (select_X1_valid, S1_Y_valid), (transform_X1_test, S1_Y_test)]

#fits the model, and enables evaluation
selection_model1.fit(select_X1_train, S1_Y_train, eval_metric = ["merror", "mlogloss"], eval_set = eval_set1, verbose = False)

# plot loss
results1 = selection_model1.evals_result()
epochs1 = len(results1['validation_0']['mlogloss'])
x_axis = range(0, epochs1)
fix, ax = pyplot.subplots()
ax.plot (x_axis, results1['validation_0']['mlogloss'], label = 'Train')
ax.plot (x_axis, results1['validation_1']['mlogloss'], label = 'Validation')
ax.plot (x_axis, results1['validation_2']['mlogloss'], label = 'Test')
ax.legend()
pyplot.ylabel('Loss')
pyplot.xlabel('epochs')
pyplot.title('Station 1 Cross-Entropy loss')
pyplot.show()

#plot classification error
fig, ax = pyplot.subplots()
ax.plot (x_axis, results1['validation_0']['merror'], label = 'Train')
ax.plot (x_axis, results1['validation_1']['merror'], label = 'Validation')
ax.plot (x_axis, results1['validation_2']['merror'], label = 'Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('epochs')
pyplot.title('Station 1 Classification Error')
pyplot.show()

#make the prediction
S1_preds = selection_model1.predict(transform_X1_test)
S1_predictions = [round(value) for value in S1_preds]
accuracyS1 = accuracy_score(S1_Y_test, S1_predictions)
print("Number of features used = {}, Station 1 Accuracy : {:.2f}%".format(select_X1_train.shape[1], accuracyS1 * 100))

#confusion matrix
cmS1 = confusion_matrix(S1_Y_test, S1_preds)
print("Station 1 Confusion Matrix\n",cmS1)

#saves the model for later
pickle_out1 = open('s1.pickle', 'wb')
pickle.dump(selection_model1, pickle_out1)
pickle_out1.close()

print("-----------------------------------------------------------------------------")    

#Setting up the SelectFromModel
selection2 = SFM(model2, threshold = 0, prefit = True)

#cuts out the unnecessary features from the dataset
select_X2_train = selection2.transform(S2_X_train)

#sets the model and tuning parameters to be used on the feature selected dataset  
selection_model2 = xgb(n_estimators = 40)

#makes a new testset sans the unnecessary features
select_X2_valid = selection2.transform(S2_X_valid)
transform_X2_test = selection2.transform(S2_X_test)

#sets up the evaluation sets to gauge model performance
eval_set2 = [(select_X2_train, S2_Y_train), (select_X2_valid, S2_Y_valid), (transform_X2_test, S2_Y_test)]

#fits the model, and enables evaluation
selection_model2.fit(select_X2_train, S2_Y_train, eval_metric = ["merror", "mlogloss"], eval_set = eval_set2, verbose = False)

# plot multiclass loss
results2 = selection_model2.evals_result()
epochs2 = len(results2['validation_0']['mlogloss'])
x_axis = range(0, epochs2)
fix, ax = pyplot.subplots()
ax.plot (x_axis, results2['validation_0']['mlogloss'], label = 'Train')
ax.plot (x_axis, results2['validation_1']['mlogloss'], label = 'Validation')
ax.plot (x_axis, results2['validation_2']['mlogloss'], label = 'Test')
ax.legend()
pyplot.ylabel('Loss')
pyplot.xlabel('epochs')
pyplot.title('Station 2 Cross-Entropy loss')
pyplot.show()

#plot classification error
fig, ax = pyplot.subplots()
ax.plot (x_axis, results2['validation_0']['merror'], label = 'Train')
ax.plot (x_axis, results2['validation_1']['merror'], label = 'Validation')
ax.plot (x_axis, results2['validation_2']['merror'], label = 'Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('epochs')
pyplot.title('Station 2 Classification Error')
pyplot.show()

#make the prediction
S2_preds = selection_model2.predict(transform_X2_test)
S2_predictions = [round(value) for value in S2_preds]
accuracyS2 = accuracy_score(S2_Y_test, S2_predictions)
print("Number of features used = {}, Station 2 Accuracy : {:.2f}%".format(select_X2_train.shape[1], accuracyS2 * 100))

#confusion matrix
cmS2 = confusion_matrix(S2_Y_test, S2_preds)
print("Station 2 Confusion Matrix\n",cmS2)

#saves the model for later
pickle_out2 = open('s2.pickle', 'wb')
pickle.dump(selection_model2, pickle_out2)
pickle_out2.close()

print("-----------------------------------------------------------------------------")

#Setting up the SelectFromModel
selection3 = SFM(model3, threshold = 0, prefit = True)

#cuts out the unnecessary features from the dataset
select_X3_train = selection3.transform(S3_X_train)

#sets the model and tuning parameters to be used on the feature selected dataset 
selection_model3 = xgb(n_estimators = 35)

#makes a new testset sans the unnecessary features
select_X3_valid = selection3.transform(S3_X_valid)
transform_X3_test = selection3.transform(S3_X_test)

#sets up the evaluation sets to gauge model performance
eval_set3 = [(select_X3_train, S3_Y_train), (select_X3_valid, S3_Y_valid), (transform_X3_test, S3_Y_test)]

#fits the model, and enables evaluation
selection_model3.fit(select_X3_train, S3_Y_train, eval_metric = ["merror", "mlogloss"], eval_set = eval_set3, verbose = False)

# plot multiclass error
results3 = selection_model3.evals_result()
epochs3 = len(results3['validation_0']['mlogloss'])
x_axis = range(0, epochs3)
fix, ax = pyplot.subplots()
ax.plot (x_axis, results3['validation_0']['mlogloss'], label = 'Train')
ax.plot (x_axis, results3['validation_1']['mlogloss'], label = 'Validation')
ax.plot (x_axis, results3['validation_2']['mlogloss'], label = 'Test')
ax.legend()
pyplot.ylabel('Loss')
pyplot.xlabel('epochs')
pyplot.title('Station 3 Cross-Entropy loss')
pyplot.show()

#plot classification error
fig, ax = pyplot.subplots()
ax.plot (x_axis, results3['validation_0']['merror'], label = 'Train')
ax.plot (x_axis, results3['validation_1']['merror'], label = 'Validation')
ax.plot (x_axis, results3['validation_2']['merror'], label = 'Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('epochs')
pyplot.title('Station 3 Classification Error')
pyplot.show()

#make the prediction
S3_preds = selection_model3.predict(transform_X3_test)
S3_predictions = [round(value) for value in S3_preds]
accuracyS3 = accuracy_score(S3_Y_test, S3_predictions)
print("Number of features used = {}, Station 3 Accuracy : {:.2f}%".format(select_X3_train.shape[1], accuracyS3 * 100))

#confusion matrix
cmS3 = confusion_matrix(S3_Y_test, S3_preds)
print("Station 3 Confusion Matrix\n",cmS3)

#saves the model for later
pickle_out3 = open('s3.pickle', 'wb')
pickle.dump(selection_model3, pickle_out3)
pickle_out3.close()

print("-----------------------------------------------------------------------------")

#Setting up the SelectFromModel
selection4 = SFM(model4, threshold = 0, prefit = True)

#cuts out the unnecessary features from the dataset
select_X4_train = selection4.transform(S4_X_train)

#sets the model and tuning parameters to be used on the feature selected dataset 
selection_model4 = xgb(n_estimators = 35)

#makes a new testset sans the unnecessary features
select_X4_valid = selection4.transform(S4_X_valid)
transform_X4_test = selection4.transform(S4_X_test)

#sets up the evaluation sets to gauge model performance
eval_set4 = [(select_X4_train, S4_Y_train), (select_X4_valid, S4_Y_valid), (transform_X4_test, S4_Y_test)]

#fits the model, and enables evaluation
selection_model4.fit(select_X4_train, S4_Y_train, eval_metric = ["merror", "mlogloss"], eval_set = eval_set4, verbose = False)

# plot multiclass error
results4 = selection_model4.evals_result()
epochs4 = len(results4['validation_0']['mlogloss'])
x_axis = range(0, epochs4)
fix, ax = pyplot.subplots()
ax.plot (x_axis, results4['validation_0']['mlogloss'], label = 'Train')
ax.plot (x_axis, results4['validation_1']['mlogloss'], label = 'Validation')
ax.plot (x_axis, results4['validation_2']['mlogloss'], label = 'Test')
ax.legend()
pyplot.ylabel('Loss')
pyplot.xlabel('epochs')
pyplot.title('Station 4 Cross-Entropy loss')
pyplot.show()

#plot classification error
fig, ax = pyplot.subplots()
ax.plot (x_axis, results4['validation_0']['merror'], label = 'Train')
ax.plot (x_axis, results4['validation_1']['merror'], label = 'Validation')
ax.plot (x_axis, results4['validation_2']['merror'], label = 'Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('epochs')
pyplot.title('Station 4 Classification Error')
pyplot.show()

#make the prediction
S4_preds = selection_model4.predict(transform_X4_test)
S4_predictions = [round(value) for value in S4_preds]
accuracyS4 = accuracy_score(S4_Y_test, S4_predictions)
print("Number of features used = {}, Station 4 Accuracy : {:.2f}%".format(select_X4_train.shape[1], accuracyS4 * 100))

#confusion matrix
cmS4 = confusion_matrix(S4_Y_test, S4_preds)
print("Station 4 Confusion Matrix\n",cmS4)

#saves the model for later
pickle_out4 = open('s4.pickle', 'wb')
pickle.dump(selection_model4, pickle_out4)
pickle_out4.close()

print("-----------------------------------------------------------------------------")
"""
#Shows the model accuracy based on number of features used & threshold level for all 4 stations.
for thresh in thresholdS1:
    selection1 = SFM(model1, threshold = thresh, prefit = True)
    select_X1_train = selection1.transform(S1_X_train)
    selection_model1 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model1.fit(select_X1_train, S1_Y_train)
    
    select_X1_test = selection1.transform(S1_X_valid)
    S1_pred = selection_model1.predict(select_X1_test)
    S1_predictions = [round(value) for value in S1_pred]
    accuracyS1 = accuracy_score(S1_Y_valid, S1_predictions)
    print("Thresh = %.3f, n=%d, Station 4 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X1_train.shape[1], accuracyS1 * 100))
print("-----------------------------------------------------------------------------")

for thresh in thresholdS2:
    selection2 = SFM(model2, threshold = thresh, prefit = True)
    select_X2_train = selection2.transform(S2_X_train)
    selection_model2 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model2.fit(select_X2_train, S2_Y_train)
    
    select_X2_test = selection2.transform(S2_X_valid)
    S2_pred = selection_model2.predict(select_X2_test)
    S2_predictions = [round(value) for value in S2_pred]
    accuracyS2 = accuracy_score(S2_Y_valid, S2_predictions)
    print("Thresh = %.3f, n=%d, Station 2 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X2_train.shape[1], accuracyS2 * 100))
print("-----------------------------------------------------------------------------")

for thresh in thresholdS3:
    selection3 = SFM(model3, threshold = thresh, prefit = True)
    select_X3_train = selection3.transform(S3_X_train)
    selection_model3 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model3.fit(select_X3_train, S3_Y_train)
    
    select_X3_test = selection3.transform(S3_X_valid)
    S3_pred = selection_model3.predict(select_X3_test)
    S3_predictions = [round(value) for value in S3_pred]
    accuracyS3 = accuracy_score(S3_Y_valid, S3_predictions)
    print("Thresh = %.3f, n=%d, Station 3 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X3_train.shape[1], accuracyS3 * 100))
print("-----------------------------------------------------------------------------")
for thresh in thresholdS4:
    selection4 = SFM(model4, threshold = thresh, prefit = True)
    select_X4_train = selection4.transform(S4_X_train)
    selection_model4 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model4.fit(select_X4_train, S4_Y_train)
    
    select_X4_test = selection4.transform(S4_X_valid)
    S4_pred = selection_model4.predict(select_X4_test)
    S4_predictions = [round(value) for value in S4_pred]
    accuracyS4 = accuracy_score(S4_Y_valid, S4_predictions)
    print("Thresh = %.3f, n=%d, Station 4 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X4_train.shape[1], accuracyS4 * 100))
print("-----------------------------------------------------------------------------")
"""
