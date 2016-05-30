#include "dataset.h"
#include<bits/stdc++.h>
using namespace std;

class LinearRegressor {
	double learning_rate_, convergence_factor_;
	Dataset & training_dataset_;
	vector<double> parameters_;
	double trainingHypothesisValue(int sample_number) {
		/*
		Returns the value of hypothesis for a particular training dataset sample
		based on current values of all parameters
		*/
		double hypothesis = 0.0;
		for (int feature_number = 0; feature_number < training_dataset_.depth(); feature_number++) {
			hypothesis += (parameters_[feature_number] * training_dataset_.getInput(sample_number, feature_number));
		}
		return hypothesis;
	}
public:

};