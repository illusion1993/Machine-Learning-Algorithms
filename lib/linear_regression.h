#include<bits/stdc++.h>
using namespace std;

class Dataset {
	long long number_of_samples_, number_of_features_;
	vector< vector<double> > inputs_;
	vector<double> outputs_;
public:
	Dataset();
	Dataset(long long number_of_samples, long long number_of_features);
};

Dataset::Dataset() {}
Dataset::Dataset(long long number_of_samples_, long long number_of_features_) {
	number_of_samples_ = number_of_samples;
	number_of_features_ = number_of_features_;
}