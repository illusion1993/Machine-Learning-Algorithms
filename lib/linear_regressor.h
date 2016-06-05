#include "dataset.h"
#include<bits/stdc++.h>
using namespace std;

class LinearRegressor {
	// A regressor object can be trained by only one training dataset, due to normalization issues
	Dataset * _training_dataset, * _testing_dataset;
	vector<double> _parameters;
	
	double hypothesisValue(int sample_number, bool training = true);
	void gradientDescent(double learning_rate, double allowed_rms_error);

public:
	void train(Dataset * training_dataset, double learning_rate, double allowed_rms_error);
	void test(Dataset * testing_dataset);
	void printResults();
};

double LinearRegressor::hypothesisValue(int sample_number, bool training /*=true*/) {
	/*
	Returns the value of hypothesis for a particular training dataset sample
	based on current values of all parameters
	*/
	double hypothesis = 0.0;
	Dataset * dataset_used = (training) ? _training_dataset : _testing_dataset;
	for (int feature_number = 0; feature_number <= dataset_used -> depth(); feature_number++) {
		hypothesis += (_parameters[feature_number] * dataset_used -> getInput(sample_number, feature_number));
	}
	return hypothesis;
}

void LinearRegressor::gradientDescent(double learning_rate, double allowed_rms_error) {
	double previous_rms_error = DBL_MAX, current_rms_error = DBL_MAX;
	vector<double> hypothesis_error(_training_dataset -> size());

	while (previous_rms_error > allowed_rms_error) {
		current_rms_error = 0;

		// Calculating all hypothesis errors for parameters updated in last iteration
		// Also calculating the RMS error value after last iteration
		for (int sample_number = 0; sample_number < _training_dataset -> size(); sample_number++) {
			hypothesis_error[sample_number] = hypothesisValue(sample_number) - _training_dataset -> getOutput(sample_number);
			current_rms_error += (hypothesis_error[sample_number] * hypothesis_error[sample_number]);
		}
		current_rms_error = sqrt(current_rms_error / _training_dataset -> size());
		// cout << "RMS error is: " << current_rms_error << "\n";

		// If the RMS error has actually increased due to the last iteration, halt the process
		if (current_rms_error > previous_rms_error) {
			cout << "Learning rate chosen is too high! Halting Gradient Descent.\n";
			break;
		}
		// Otherwise, current RMS error is now stored as previous RMS error
		previous_rms_error = current_rms_error;

		// Now updating the values of parameters
		// For each parameter p, the partial derivative of the cost function wrt the parameter is found by
		// Multiply the hypothesis error (h(theta) - Y) with the parameter's coefficient in that particular sample, for all samples
		// Divide the sum of all those values by the number of samples. Parameter is decremented by learning rate * this value
		for (int feature_number = 0; feature_number <= _training_dataset -> depth(); feature_number++) {
			double partial_derivative = 0;
			for (int sample_number = 0; sample_number < _training_dataset -> size(); sample_number++) {
				partial_derivative += (hypothesis_error[sample_number] * _training_dataset -> getInput(sample_number, feature_number));
				partial_derivative /= _training_dataset -> size();
				_parameters[feature_number] -= (learning_rate * partial_derivative);
			}
		}
	}
}

void LinearRegressor::train(Dataset * training_dataset, double learning_rate, double allowed_rms_error) {
	_training_dataset = training_dataset;
	_parameters.resize(_training_dataset -> depth() + 1, 0.5);
	gradientDescent(learning_rate, allowed_rms_error);
}

void LinearRegressor::test(Dataset * testing_dataset) {
	_testing_dataset = testing_dataset;
	for (int sample_number = 0; sample_number < _testing_dataset -> size(); sample_number++) {
		cout << "Testing result: " << hypothesisValue(sample_number, false) << "\n";
	}
}

void LinearRegressor::printResults() {
	cout << "Final parameters:\n";
	for (int i = 0; i < _training_dataset -> depth() + 1; i++) cout << _parameters[i] << " ";
	cout << "\n\nHypotheses:\n";
	for (int i = 0; i < _training_dataset -> size(); i++) {
		for (int j = 0; j < _training_dataset -> depth() + 1; j++) {
			cout << _training_dataset -> getInput(i, j) << " ";
		}
		cout << ": " << _training_dataset -> getOutput(i) << ", h(X) = " << hypothesisValue(i) << "\n";
	}
}