# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:45:55 2020

@author: VICTOR
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


#""" First reading the fully packed excel file to extract the needed dataset (Existing Employee and Ex Employee)."""
#
#excel1 = pd.read_excel('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', sheet_name= 'Existing employees') 
#excel2 = pd.read_excel('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', sheet_name= 'Employees who have left')
#excel1.to_excel('Existing Employees.xlsx') # Saving the read sheet as a excel file
#excel2.to_excel('Ex Employees.xlsx')        # Saving the read sheet as a excel file


""" Now to read each excel file saved (Existing Employees & Ex Employees) """

ExistingEmployees= pd.read_excel('Existing Employees.xlsx')
ExEmployees= pd.read_excel('Ex Employees.xlsx')

print(ExistingEmployees)
print(ExEmployees)

""" Slicing out values of each dataset """

first = ExistingEmployees.iloc[:,0:22]
second = ExEmployees.iloc[:, 0:22]

""" Encoding and Concatenation"""

jointhem = DataFrame(np.concatenate((first,second)))

from sklearn.preprocessing import LabelEncoder

for i in jointhem.columns:
    if jointhem[i].dtype==np.number:
        continue
    jointhem[i]=LabelEncoder().fit_transform(jointhem[i])


#print(jointhem) # This was commented to save time of program run.

""" Splitting the data into Independent (X) and dependent variable (Y) """

depvar = jointhem.iloc[:, 0:11]      # That is, dependent variable
indvar = jointhem.iloc[:,3]         # That is, independent variable.

# Now to Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(depvar,indvar, test_size=0.20,random_state=0)



# Using Random Forest to check accuracy of the data.
from sklearn.ensemble import RandomForestClassifier
forest =  RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state =0)
forest.fit(X_train, Y_train)
print('Accuracy of Random Forest in the data model =',forest.score(X_train, Y_train)*100) # The 100 is for conversion to percentage.


""" LINEAR REGRESSION -  To predict what types of workers are leaving."""
# Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting Test set results
LinearRegressionPredictions = regressor.predict(X_test)
LinearRegressionPredictions = pd.DataFrame(data = LinearRegressionPredictions)


""" LOGISTIC REGRESSION -  To predict/check if a worker will leave or not."""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='lbfgs')
classifier.fit(X_train, Y_train)

# Predicting Test set results
LogisticRegressionPredictions = classifier.predict(X_test)


var_prob = classifier.predict_proba(X_test)
var_prob[0, :]

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,LogisticRegressionPredictions)


""" Adding the predicted values in the main dataframe. That is, the predictions of employees
 that are most likely to leave the company."""
 
LogisticRegressionPredictions = pd.DataFrame(data = LogisticRegressionPredictions)
Pronetoleave_OriginalDataFrame = pd.merge(jointhem,LogisticRegressionPredictions, how ='left',left_index =True, right_index =True) # Predictions of workers prone to leave merged with original DataFrame of all.

Employee_Prone_to_Leave = Pronetoleave_OriginalDataFrame.loc[Pronetoleave_OriginalDataFrame[3]==any(LogisticRegressionPredictions)]
Employee_Prone_to_Leave.rename(columns={'1':'Emp ID'}, inplace =True)


""" VISUALIZATION """
plt.title('Employees prone to leave.')
plt.hist(Employee_Prone_to_Leave, bins = 10)
plt.xlabel('Emp ID')
plt.ylabel('Predictions')
plt.savefig('Employees prone to leave.')
plt.show()

""" Saving predictions (Linear and Logistic Regression) to document files """
LinearRegressionPredictions.to_excel('Linear Regression Predictions.xlsx')
LogisticRegressionPredictions.to_excel('Logistic Regression Predictions.xlsx')
Employee_Prone_to_Leave.to_excel('Employees that are prone to leave.xlsx') # Finally,Saving the employees that are prone to leave in a document (excel file) 




















