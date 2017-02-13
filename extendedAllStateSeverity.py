import pandas as pd  
import numpy as np

from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost
from xgboost import XGBRegressor

import time



def main():
	train_data = pd.read_csv('train.csv', index_col = 0)
	test_data = pd.read_csv('test.csv', index_col = 0)

	seed = 0
	alpha = 1
	split = 116
	val_size = 0.1
	results = {}

	print('Kindly select the type of transformation you would like to run:')
	print('1. Regular Encoding Method')
	print('2. One Hot Encoding Method')
	print('Please select the option (1/2)')

	choice = int(input())

	if choice==1:
		start_time = time.clock()
		print('Executing Regular Tranformation....')
		collection = dict()
		count = 0
		n = 130

		for i in range(1,117):
			temp_col = list(train_data["cat"+str(i)])
			for j in range(0,len(temp_col)):
				val = temp_col[j]
				if val in collection:
					temp_col[j] = collection[val]
				else:
					count+=1
					collection[val] = count
					temp_col[j] = collection[val]
			train_data["cat"+str(i)] = temp_col

		train, test = splitData(train_data, seed)
		#gradient_boosting(train, test, seed, n)
		xgboost(train, test, seed, n)


	elif choice==2:
		start_time = time.clock()
		df, n = oneHotEncoding(train_data, split)
		train, test = splitData(df, seed)
		#lasso_regression(train, test, seed, alpha, n)
		xgboost(train, test, seed, n)
	

	print('--------', time.clock()-start_time, 'seconds-------')
	

def oneHotEncoding(train_data, split):
	print('Executing One Hot Encoding....')
	labels = []
	cols = train_data.columns
	index = train_data.index

	for i in range(0,split):
		train = train_data[cols[i]].unique()
		labels.append(list(set(train)))

	cats = []
	for i in range(0, split):
		label_encoder = LabelEncoder()
		label_encoder.fit(labels[i])
		feature = label_encoder.transform(train_data.iloc[:,i])
		feature = feature.reshape(train_data.shape[0], 1)
		onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
		feature = onehot_encoder.fit_transform(feature)
		cats.append(feature)

	# Make a 2D array from a list of 1D arrays
	encoded_cats = np.column_stack(cats)

	#Concatenate encoded attributes with continuous attributes
	dataset_encoded = np.concatenate((encoded_cats,train_data.iloc[:,split:].values),axis=1)
	del cats
	del feature
	del train_data
	del encoded_cats
	#print(dataset_encoded.shape)
	r,c = dataset_encoded.shape
	i_cols = []
	for i in range(0,c-1):
		i_cols.append(i)

	X = dataset_encoded[:,0:(c-1)]
	Y = dataset_encoded[:,(c-1)]
	del dataset_encoded
	#print(X, Y)
	r1, c1 = X.shape
	df = pd.DataFrame(X, index = index , columns = [x for x in range(0,c1)])
	df['loss'] = Y
	del X, Y
	m, n = df.shape
	return df, n-1

def kfoldvalidation(X, Y , val_size, seed):
	X_train, Y_train, X_val, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
	del X
	del Y
	return (X_train, X_val, Y_train, Y_val)

def splitData(df, seed):
	print('Splitting data....')
	train_data , test_data = train_test_split(df, test_size=0.2, random_state = seed)
	print("Train data size: "+str(len(train_data)))
	print("Test data size: "+str(len(test_data)))
	return train_data, test_data


def linear_regression(train, test, n):
	model = LinearRegression(n_jobs=-1).fit(train.iloc[:,:n], train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'], predicted)
	print("Linear Regression: "+str(round(result,2)))

	
def lasso_regression(train, test, seed, alpha, n):
	print('Executing Lasso Regression......(Processing.....)')
	model = Lasso(alpha = alpha,random_state = seed).fit(train.iloc[:,:n],train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'],predicted)
	res_df = pd.DataFrame({'Index' : list(test.index), 'Loss' : list(predicted)}, columns = ['Index', 'Loss'])
	res_df.to_csv('Predicted_loss_values_OneHotEncoding.csv', index = False)
	print("Lasso Regression: "+str(round(result,2)))
	print("Check your folder for the file 'Predicted_loss_values.csv' to see the final predicted results for the Test Data")
	

def elastic_net_regression(test, train, seed, alpha, n):
	model = ElasticNet(alpha= alpha, random_state = seed).fit(train.iloc[:,:n],train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'],predicted)
	print("Elastic Net: "+str(round(result,2)))


def ridge_regression(test, train, seed, alpha, n):
	model = Ridge(alpha = alpha, random_state = seed).fit(train.iloc[:,:n],train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'],predicted)
	print("Ridge Regresssion: "+str(round(result,2)))
	

def kneighbors(test, train, n):
	model = KNeighborsRegressor(n_neighbors = 8 , n_jobs = -1).fit(train.iloc[:,:n],train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'],predicted)
	print("KNeighbors: "+str(round(result,2)))
	

def adaboost(test, train, seed, n):
	model = AdaBoostRegressor(n_estimators = 100, random_state = seed).fit(train.iloc[:,:n],train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'],predicted)
	print("AdaBoost: "+str(round(result,2)))


def gradient_boosting(train, test, seed, n):
	print('Executing Gradient Boosting Regression.....(Processing.....)')
	model = GradientBoostingRegressor(n_estimators = 2000 , random_state = seed).fit(train.iloc[:,:n],train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'],predicted)
	res_df = pd.DataFrame({'Index' : list(test.index), 'Loss' : list(predicted)}, columns = ['Index', 'Loss'])
	res_df.to_csv('Predicted_loss_values_RegularTranformation.csv', index = False)
	print("Gradient Boosting: "+str(round(result,2)))
	print("Check your folder for the file 'Predicted_loss_values_RegularTransformation(2000).csv' to see the final predicted results for the Test Data")

def xgboost(train, test, seed, n):
	print('Executing XGBoost...')
	model = XGBRegressor(n_estimators = 2000, seed = seed).fit(train.iloc[:,:n], train['loss'])
	predicted = model.predict(test.iloc[:,:n])
	result = mean_absolute_error(test['loss'], predicted)
	res_df = pd.DataFrame({'Index' : list(test.index), 'Loss' : list(predicted)}, columns = ['Index', 'Loss'])
	res_df.to_csv('Predicted_loss_values_RegularTranformation.csv', index = False)
	print('XGRegressor:'+str(round(result,2)))
	print("Check your folder for the file 'Predicted_loss_values.csv' to see the final predicted results for the Test Data")

if __name__ == '__main__':
	main()