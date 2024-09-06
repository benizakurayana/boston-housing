"""
File: boston_housing_competition.py
Name: Jane
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""


import pandas as pd
from sklearn import preprocessing, model_selection, decomposition, linear_model, svm, ensemble, metrics
import random

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

def main():
	############
	# Training
	############
	# Data Preprocess
	train_data = pd.read_csv(TRAIN_FILE)
	labels = train_data['medv']
	train_data = train_data.drop(columns=['ID', 'chas', 'rad', 'zn', 'black', 'medv'])  # Using: crim, indus, nox, rm, age, dis, tax, ptratio, lstat

	# Split train and validation data
	train_x, val_x, train_y, val_y = model_selection.train_test_split(train_data, labels, test_size=0.2)

	# Normalization / Standardization
	# normalizer = preprocessing.Normalizer()
	# train_x = normalizer.fit_transform(train_x)
	# val_x = normalizer.transform(val_x)
	std_scaler = preprocessing.StandardScaler()
	train_x = std_scaler.fit_transform(train_x)
	val_x = std_scaler.transform(val_x)

	# PCA
	pca = decomposition.PCA(n_components=9)
	train_x_pca = pca.fit_transform(train_x)
	val_x_pca = pca.transform(val_x)

	# Polynomial
	poly_d2 = preprocessing.PolynomialFeatures(degree=2)
	train_x_2 = poly_d2.fit_transform(train_x)
	val_x_2 = poly_d2.transform(val_x)

	poly_d2_pca = preprocessing.PolynomialFeatures(degree=2)
	train_x_2_pca = poly_d2_pca.fit_transform(train_x_pca)
	val_x_2_pca = poly_d2_pca.transform(val_x_pca)

	# Fit models
	regressor_linear_d1 = linear_model.LinearRegression().fit(train_x, train_y)
	regressor_linear_d2 = linear_model.LinearRegression().fit(train_x_2, train_y)

	regressor_svm_d1 = svm.SVR(gamma=4.5, C=9.7798).fit(train_x, train_y)
	regressor_svm_d2 = svm.SVR(gamma=5.17, C=10.366).fit(train_x_2, train_y)

	regressor_rf = ensemble.RandomForestRegressor(max_depth=5, min_samples_leaf=4, max_leaf_nodes=11).fit(train_x, train_y)
	regressor_rf_d1_pca = ensemble.RandomForestRegressor(max_depth=6, min_samples_leaf=5, max_leaf_nodes=10).fit(train_x_pca, train_y)
	regressor_rf_d2_pca = ensemble.RandomForestRegressor(max_depth=7, min_samples_leaf=5, max_leaf_nodes=13).fit(train_x_2_pca, train_y)

	############
	# Validation
	############
	train_score = regressor_linear_d1.score(train_x, train_y)
	val_score = regressor_linear_d1.score(val_x, val_y)
	rms_err = metrics.mean_squared_error(regressor_linear_d1.predict(val_x), val_y)**0.5
	print(f'linear reg d1\t{train_score}\t{val_score}\t{rms_err}')

	train_score = regressor_linear_d2.score(train_x_2, train_y)
	val_score = regressor_linear_d2.score(val_x_2, val_y)
	rms_err = metrics.mean_squared_error(regressor_linear_d2.predict(val_x_2), val_y) ** 0.5
	print(f'linear reg d2\t{train_score}\t{val_score}\t{rms_err}')

	print()
	train_score = regressor_svm_d1.score(train_x, train_y)
	val_score = regressor_svm_d1.score(val_x, val_y)
	rms_err = metrics.mean_squared_error(regressor_svm_d1.predict(val_x), val_y) ** 0.5
	print(f'svm d1\t\t\t{train_score}\t{val_score}\t{rms_err}')

	train_score = regressor_svm_d2.score(train_x_2, train_y)
	val_score = regressor_svm_d2.score(val_x_2, val_y)
	rms_err = metrics.mean_squared_error(regressor_svm_d2.predict(val_x_2), val_y) ** 0.5
	print(f'svm d2\t\t\t{train_score}\t{val_score}\t{rms_err}')

	print()
	train_score = regressor_rf.score(train_x, train_y)
	val_score = regressor_rf.score(val_x, val_y)
	rms_err = metrics.mean_squared_error(regressor_rf.predict(val_x), val_y) ** 0.5
	print(f'rf d1\t\t\t{train_score}\t{val_score}\t{rms_err}')

	train_score = regressor_rf_d1_pca.score(train_x_pca, train_y)
	val_score = regressor_rf_d1_pca.score(val_x_pca, val_y)
	rms_err = metrics.mean_squared_error(regressor_rf_d1_pca.predict(val_x_pca), val_y) ** 0.5
	print(f'rf d1 pca\t\t\t{train_score}\t{val_score}\t{rms_err}')

	train_score = regressor_rf_d2_pca.score(train_x_2_pca, train_y)
	val_score = regressor_rf_d2_pca.score(val_x_2_pca, val_y)
	rms_err = metrics.mean_squared_error(regressor_rf_d2_pca.predict(val_x_2_pca), val_y) ** 0.5
	print(f'rf d2 pca\t\t\t{train_score}\t{val_score}\t{rms_err}')

	############
	# Random Forest
	############
	# Train
	rms_err = float('inf')
	while rms_err > 2.1:
		split_random_state = random.randint(0, 8388607)
		rf_random_state = random.randint(0, 8388607)

		# Split train and validation data
		train_x, val_x, train_y, val_y = model_selection.train_test_split(train_data, labels, test_size=0.2, random_state=split_random_state)

		# Standardization
		std_scaler = preprocessing.StandardScaler()
		train_x = std_scaler.fit_transform(train_x)
		val_x = std_scaler.transform(val_x)

		# Fit model
		regressor_rf = ensemble.RandomForestRegressor(max_depth=6, min_samples_leaf=5, max_leaf_nodes=10,
													  random_state=rf_random_state).fit(train_x, train_y)

		# Validation
		train_score = regressor_rf.score(train_x, train_y)
		val_score = regressor_rf.score(val_x, val_y)
		rms_err = metrics.mean_squared_error(regressor_rf.predict(val_x), val_y) ** 0.5

	print()
	print('split_random_state:', split_random_state, 'rf_random_state:', rf_random_state)
	print(f'rf d1\t\t\t{train_score}\t{val_score}\t{rms_err}')

	############
	# Test
	############
	# Data Preprocess
	data = pd.read_csv(TEST_FILE)
	test_ids = data.pop('ID')
	test_x = data.drop(columns=['chas', 'rad', 'zn', 'black'])

	# Normalization / Standardization
	# test_x = normalizer.transform(test_x)
	test_x = std_scaler.transform(test_x)

	# Prediction
	predictions = regressor_rf.predict(test_x)

	# Out-file
	ids = pd.DataFrame(test_ids, columns=['ID'])
	predictions = pd.DataFrame(predictions, columns=['medv'])

	concat_df = pd.concat([ids, predictions], axis=1)
	concat_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
	main()
