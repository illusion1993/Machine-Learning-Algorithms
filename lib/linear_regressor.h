#include "dataset.h"
#include<bits/stdc++.h>
using namespace std;

class LinearRegressor {
	// A regressor object can be trained by only one training dataset, due to normalization issues
	Dataset * training_dataset_;
	vector<double> parameters_;
	
	double trainingHypothesisValue(int sample_number) {
		/*
		Returns the value of hypothesis for a particular training dataset sample
		based on current values of all parameters
		*/
		double hypothesis = 0.0;
		for (int feature_number = 0; feature_number <= training_dataset_ -> depth(); feature_number++) {
			hypothesis += (parameters_[feature_number] * training_dataset_ -> getInput(sample_number, feature_number));
		}
		return hypothesis;
	}
	
	void gradientDescent(double learning_rate, double allowed_rms_error) {
		double previous_rms_error = DBL_MAX, current_rms_error = DBL_MAX;
		vector<double> hypothesis_error(training_dataset_ -> size());

		while (previous_rms_error > allowed_rms_error) {
			current_rms_error = 0;

			// Calculating all hypothesis errors for parameters updated in last iteration
			// Also calculating the RMS error value after last iteration
			for (int sample_number = 0; sample_number < training_dataset_ -> size(); sample_number++) {
				hypothesis_error[sample_number] = trainingHypothesisValue(sample_number) - training_dataset_ -> getOutput(sample_number);
				current_rms_error += (hypothesis_error[sample_number] * hypothesis_error[sample_number]);
			}
			current_rms_error = sqrt(current_rms_error / training_dataset_ -> size());
			cout << "RMS error is: " << current_rms_error << "\n";

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
			for (int feature_number = 0; feature_number <= training_dataset_ -> depth(); feature_number++) {
				double partial_derivative = 0;
				for (int sample_number = 0; sample_number < training_dataset_ -> size(); sample_number++) {
					partial_derivative += (hypothesis_error[sample_number] * training_dataset_ -> getInput(sample_number, feature_number));
					partial_derivative /= training_dataset_ -> size();
					parameters_[feature_number] -= (learning_rate * partial_derivative);
				}
			}
		}
	}
public:
	void train(Dataset * training_dataset, double learning_rate, double allowed_rms_error) {
		training_dataset_ = training_dataset;
		parameters_.resize(training_dataset_ -> depth() + 1, 0.5);
		gradientDescent(learning_rate, allowed_rms_error);
	}

	void printResults() {
		cout << "Final parameters:\n";
		for (int i = 0; i < training_dataset_ -> depth() + 1; i++) cout << parameters_[i] << " ";
		cout << "\n\nHypotheses:\n";
		for (int i = 0; i < training_dataset_ -> size(); i++) {
			for (int j = 0; j < training_dataset_ -> depth() + 1; j++) {
				cout << training_dataset_ -> getInput(i, j) << " ";
			}
			cout << ": " << training_dataset_ -> getOutput(i) << ", h(X) = " << trainingHypothesisValue(i) << "\n";
		}
	}
};