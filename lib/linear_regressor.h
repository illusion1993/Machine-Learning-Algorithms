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
		double rms_error = allowed_rms_error + 1;
		vector<double> hypothesis_error(training_dataset_ -> size());

		while (rms_error > allowed_rms_error) {
			rms_error = 0;
			for (int sample_number = 0; sample_number < training_dataset_ -> size(); sample_number++) {
				hypothesis_error[sample_number] = trainingHypothesisValue(sample_number) - training_dataset_ -> getOutput(sample_number);
				rms_error += (hypothesis_error[sample_number] * hypothesis_error[sample_number]);
			}
			rms_error = sqrt(rms_error / training_dataset_ -> size());
			for (int feature_number = 0; feature_number <= training_dataset_ -> depth(); feature_number++) {
				double partial_derivative = 0;
				for (int sample_number = 0; sample_number < training_dataset_ -> size(); sample_number++) {
					partial_derivative += (hypothesis_error[sample_number] * training_dataset_ -> getInput(sample_number, feature_number));
					partial_derivative /= training_dataset_ -> size();
					parameters_[feature_number] -= (learning_rate * partial_derivative);
				}
			}
			cout << "RMS error is: " << rms_error << "\n";
		}
	}
public:
	void train(Dataset * training_dataset, double learning_rate, double convergence_factor) {
		training_dataset_ = training_dataset;
		parameters_.resize(training_dataset_ -> depth() + 1, 0.5);
		gradientDescent(learning_rate, convergence_factor);
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