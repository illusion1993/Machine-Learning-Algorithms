#include<bits/stdc++.h>
using namespace std;

class Dataset {
	int size_, depth_;
	vector< vector<double> > inputs_;
	vector<double> outputs_;
	bool has_outputs_;
	void addBiasInput();
	void reset(int, int, bool);
public:
	Dataset();
	Dataset(int, int, bool);
	Dataset(std::ifstream&, int, int, bool);
	void readDatasetFile(std::ifstream&, int, int, bool);
	void printDataset();
};

Dataset::Dataset() {}
Dataset::Dataset(int size, int depth, bool has_outputs) { reset(size, depth, has_outputs); }
Dataset::Dataset(std::ifstream &fin, int size, int depth, bool has_outputs) { readDatasetFile(fin, size, depth, has_outputs); }
void Dataset::reset(int size, int depth, bool has_outputs) {
	size_ = size;
	depth_ = depth;
	has_outputs_ = has_outputs;
	inputs_.resize(size, vector<double>(depth));
	if (has_outputs) outputs_.resize(size);
}
void Dataset::readDatasetFile(std::ifstream &fin, int size, int depth, bool has_outputs) {
	reset(size, depth, has_outputs);
	// Clear previous data, this may be called to load another dataset file
	for (int sample_number = 0; sample_number < size_; sample_number++) {
		for (int feature_number = 0; feature_number < depth_; feature_number++) {
			fin >> inputs_[sample_number][feature_number];
		}
		if (has_outputs_) {
			fin >> outputs_[sample_number];
		}
	}
	// After reading and storing dataset, add a bias input (1.0)
	addBiasInput();
}
void Dataset::addBiasInput() {
	// Do not proceed if already added a bias input
}
void Dataset::printDataset() {
	cout << "size: " << size_ << "\n";
	cout << "depth: " << depth_ << "\n";
	cout << "inputs vector sizes: " << inputs_.size() << ", " << inputs_[0].size() << "\n";
	cout << "output vector size: " << outputs_.size() << "\n";
	cout << "has outputs: " << has_outputs_ << "\n";
	cout << "Values in samples are: \n";
	for (int sample_number = 0; sample_number < size_; sample_number++) {
		for (int feature_number = 0; feature_number < depth_; feature_number++) {
			cout << inputs_[sample_number][feature_number] << " ";
		}
		if (has_outputs_) cout << ": " << outputs_[sample_number] << "\n";
	}
}