#include<bits/stdc++.h>
using namespace std;

class Dataset {
	// Size of dataset is number of samples, depth is number of features
	int size_, depth_;
	
	// Inputs, outputs and their normalized versions for all samples in dataset
	vector< vector<double> > inputs_, normalized_inputs_;
	vector<double> outputs_, normalized_outputs_;
	
	// Maximum and minimum values of inputs/outputs in dataset. Used for normalization
	vector<double> maximum_input_, minimum_input_;
	double maximum_output_, minimum_output_;
	
	// has_outputs would be true for training dataset, false for a testing dataset
	// is_inflated is true when vectors are inflated to required sizes
	// is_populated is true when dataset has loaded values from file
	bool has_outputs_, is_normalized_, is_inflated, is_populated_;
	
	void reset(int size = 0, int depth = 0, bool has_outputs = false);
public:
	Dataset();
	Dataset(int size, int depth, bool has_outputs);
	Dataset(std::ifstream &fin, int size, int depth, bool has_outputs);
	
	void readDatasetFile(std::ifstream &fin, int size, int depth, bool has_outputs);
	void normalize();
	int size();
	int depth();
	bool hasOutputs();
	double getInput(int sample_number, int feature_number);
	double getOutput(int sample_number);

	void printDataset();
};

Dataset::Dataset() { reset(); }

Dataset::Dataset(int size, int depth, bool has_outputs) { reset(size, depth, has_outputs); }

Dataset::Dataset(std::ifstream &fin, int size, int depth, bool has_outputs) { readDatasetFile(fin, size, depth, has_outputs); }

void Dataset::reset(int size /*=0*/, int depth /*=0*/, bool has_outputs /*=false*/) {
	size_ = size;
	depth_ = depth;
	has_outputs_ = has_outputs;
	
	inputs_.resize(size_, vector<double>(depth_));
	if (has_outputs_) outputs_.resize(size_);
	
	normalized_inputs_.clear();
	normalized_outputs_.clear();

	maximum_output_ = DBL_MIN;
	minimum_output_ = DBL_MAX;
	maximum_input_.resize(depth, DBL_MIN);
	minimum_input_.resize(depth, DBL_MAX);

	is_normalized_ = is_populated_ = false;
	is_inflated = (size && depth);
}

void Dataset::readDatasetFile(std::ifstream &fin, int size, int depth, bool has_outputs) {
	reset(size, depth, has_outputs);
	for (int sample_number = 0; sample_number < size_; sample_number++) {
		for (int feature_number = 0; feature_number < depth_; feature_number++) {
			fin >> inputs_[sample_number][feature_number];
			maximum_input_[feature_number] = max(maximum_input_[feature_number], inputs_[sample_number][feature_number]);
			minimum_input_[feature_number] = min(minimum_input_[feature_number], inputs_[sample_number][feature_number]);
		}
		if (has_outputs_) {
			fin >> outputs_[sample_number];
			maximum_output_ = max(maximum_output_, outputs_[sample_number]);
			minimum_output_ = min(minimum_output_, outputs_[sample_number]);
		}
		// Adding bias input (1.0) to every sample
		inputs_[sample_number].push_back(1.0);
	}
	is_populated_ = true;
}

void Dataset::normalize() {
	// Checking 1. is not already normalized, 2. dataset is populated
	// 3. has_outputs i.e. This is a training dataset. The testing dataset should not be normalized here.
	if (!is_normalized_ && is_populated_ && has_outputs_) {
		normalized_inputs_.resize(size_, vector<double>(depth_));
		normalized_outputs_.resize(size_);
		// Min-Max feature scaling in action
		for (int sample_number = 0; sample_number < size_; sample_number++) {
			for (int feature_number = 0; feature_number < depth_; feature_number++) {
				normalized_inputs_[sample_number][feature_number] = 
					(maximum_input_[feature_number] - minimum_input_[feature_number] > 0.0) ? 
						(inputs_[sample_number][feature_number] - minimum_input_[feature_number]) / (maximum_input_[feature_number] - minimum_input_[feature_number]) : 1;
			}
			normalized_outputs_[sample_number] = (maximum_output_ - minimum_output_ > 0.0) ? 
				(outputs_[sample_number] - minimum_output_) / (maximum_output_ - minimum_output_) : 1;
			// Pushing bias input in the end
			normalized_inputs_[sample_number].push_back(1.0);
		}
		is_normalized_ = true;
	}
}

int Dataset::size() { return size_; }

int Dataset::depth() { return depth_; }

bool Dataset::hasOutputs() { return has_outputs_; }

double Dataset::getInput(int sample_number, int feature_number) {
	return (is_normalized_) ? normalized_inputs_[sample_number][feature_number] : inputs_[sample_number][feature_number];
}

double Dataset::getOutput(int sample_number) {
	return (is_normalized_) ? normalized_outputs_[sample_number] : outputs_[sample_number];
}

void Dataset::printDataset() {
	// This is helpful for debugging purposes
	cout << "\n\nDataset Description------------\nSize: " << size_ << "\n";
	cout << "Depth: " << depth_ << "\n";
	cout << "Inputs vector sizes: " << inputs_.size() << ", " << inputs_[0].size() << "\n";
	cout << "Outputs vector size: " << outputs_.size() << "\n";
	cout << "has_outputs: " << has_outputs_ << "\n";
	cout << "is_normalized: " << is_normalized_ << "\n";
	cout << "is_populated: " << is_populated_ << "\n\n";
	if (is_inflated) {
		cout << "Values in samples are: \n";
		for (int sample_number = 0; sample_number < size_; sample_number++) {
			for (int feature_number = 0; feature_number <= depth_; feature_number++) {
				cout << inputs_[sample_number][feature_number] << "\t";
			}
			if (has_outputs_) cout << ": " << outputs_[sample_number] << "\n";
		}
	}
	if (is_normalized_) {
		cout << "\nNormalized values: \n";
		for (int sample_number = 0; sample_number < size_; sample_number++) {
			for (int feature_number = 0; feature_number <= depth_; feature_number++) {
				cout << normalized_inputs_[sample_number][feature_number] << "\t";
			}
			if (has_outputs_) cout << ": " << normalized_outputs_[sample_number] << "\n";
		}
	}
	cout << "\nMaximum Inputs for each feature: ";
	for (int feature_number = 0; feature_number < depth_; feature_number++) {
		cout << maximum_input_[feature_number] << " ";
	}
	cout << "\nMinimum Inputs for each feature: ";
	for (int feature_number = 0; feature_number < depth_; feature_number++) {
		cout << minimum_input_[feature_number] << " ";
	}
	cout << "\nMaximum Output: " << maximum_output_ << "\n";
	cout << "Minimum Output: " << minimum_output_ << "\n";
}