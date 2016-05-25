#include<bits/stdc++.h>
using namespace std;

class Dataset {
	int size_, depth_;
	vector< vector<double> > inputs_;
	vector<double> outputs_;

	void addBiasInput();
public:
	Dataset();
	Dataset(int, int);
	Dataset(std::ifstream&, int, int);

	void readDatasetFile(std::ifstream&, int, int);
};

Dataset::Dataset() {}
Dataset::Dataset(int size, int depth) : size_(size), depth_(depth) {
	inputs_.resize(size, vector<double>(depth));
	outputs_.resize(size);
}
Dataset::Dataset(std::ifstream &fin, int size, int depth) : size_(size), depth_(depth) {
	inputs_.resize(size, vector<double>(depth));
	outputs_.resize(size);
	readDatasetFile(fin, size, depth);
}
void Dataset::readDatasetFile(std::ifstream &fin, int size, int depth) : size_(size), depth_(depth) {
	// Clear previous data, this may be called to load another dataset file

	// After reading and storing dataset, add a bias input (1.0)
	addBiasInput();
}
void Dataset::addBiasInput() {
	// Do not proceed if already added a bias input
}