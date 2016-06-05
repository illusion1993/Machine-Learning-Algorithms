#include<bits/stdc++.h>
using namespace std;

class Dataset {
	// Size of dataset is number of samples, depth is number of features
	int _size, _depth;
	
	// Inputs, outputs and their normalized versions for all samples in dataset
	vector< vector<double> > _inputs, _normalized_inputs;
	vector<double> _outputs, _normalized_outputs;
	
	// Maximum and minimum values of inputs/outputs in dataset. Used for normalization
	vector<double> _maximum_input, _minimum_input;
	double _maximum_output, _minimum_output;
	
	// _has_outputs would be true for training dataset, false for a testing dataset
	// _is_inflated is true when vectors are inflated to required sizes
	// _is_populated is true when dataset has loaded values from file
	bool _has_outputs, _is_normalized, _is_inflated, _is_populated;
	
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
	_size = size;
	_depth = depth;
	_has_outputs = has_outputs;
	
	_inputs.resize(_size, vector<double>(_depth));
	if (_has_outputs) _outputs.resize(_size);
	
	_normalized_inputs.clear();
	_normalized_outputs.clear();

	_maximum_output = DBL_MIN;
	_minimum_output = DBL_MAX;
	_maximum_input.resize(depth, DBL_MIN);
	_minimum_input.resize(depth, DBL_MAX);

	_is_normalized = _is_populated = false;
	_is_inflated = (size && depth);
}

void Dataset::readDatasetFile(std::ifstream &fin, int size, int depth, bool has_outputs) {
	reset(size, depth, has_outputs);
	for (int sample_number = 0; sample_number < _size; sample_number++) {
		for (int feature_number = 0; feature_number < _depth; feature_number++) {
			fin >> _inputs[sample_number][feature_number];
			_maximum_input[feature_number] = max(_maximum_input[feature_number], _inputs[sample_number][feature_number]);
			_minimum_input[feature_number] = min(_minimum_input[feature_number], _inputs[sample_number][feature_number]);
		}
		if (_has_outputs) {
			fin >> _outputs[sample_number];
			_maximum_output = max(_maximum_output, _outputs[sample_number]);
			_minimum_output = min(_minimum_output, _outputs[sample_number]);
		}
		// Adding bias input (1.0) to every sample
		_inputs[sample_number].push_back(1.0);
	}
	_is_populated = true;
}

void Dataset::normalize() {
	// Checking 1. is not already normalized, 2. dataset is populated
	// 3. has_outputs i.e. This is a training dataset. The testing dataset should not be normalized here.
	if (!_is_normalized && _is_populated && _has_outputs) {
		_normalized_inputs.resize(_size, vector<double>(_depth));
		_normalized_outputs.resize(_size);
		// Min-Max feature scaling in action
		for (int sample_number = 0; sample_number < _size; sample_number++) {
			for (int feature_number = 0; feature_number < _depth; feature_number++) {
				_normalized_inputs[sample_number][feature_number] = 
					(_maximum_input[feature_number] - _minimum_input[feature_number] > 0.0) ? 
						(_inputs[sample_number][feature_number] - _minimum_input[feature_number]) / (_maximum_input[feature_number] - _minimum_input[feature_number]) : 1;
			}
			_normalized_outputs[sample_number] = (_maximum_output - _minimum_output > 0.0) ? 
				(_outputs[sample_number] - _minimum_output) / (_maximum_output - _minimum_output) : 1;
			// Pushing bias input in the end
			_normalized_inputs[sample_number].push_back(1.0);
		}
		_is_normalized = true;
	}
}

int Dataset::size() { return _size; }

int Dataset::depth() { return _depth; }

bool Dataset::hasOutputs() { return _has_outputs; }

double Dataset::getInput(int sample_number, int feature_number) {
	return (_is_normalized) ? _normalized_inputs[sample_number][feature_number] : _inputs[sample_number][feature_number];
}

double Dataset::getOutput(int sample_number) {
	return (_is_normalized) ? _normalized_outputs[sample_number] : _outputs[sample_number];
}

void Dataset::printDataset() {
	// This is helpful for debugging purposes
	cout << "\n\nDataset Description------------\nSize: " << _size << "\n";
	cout << "Depth: " << _depth << "\n";
	cout << "Inputs vector sizes: " << _inputs.size() << ", " << _inputs[0].size() << "\n";
	cout << "Outputs vector size: " << _outputs.size() << "\n";
	cout << "has_outputs: " << _has_outputs << "\n";
	cout << "is_normalized: " << _is_normalized << "\n";
	cout << "is_populated: " << _is_populated << "\n\n";
	if (_is_inflated) {
		cout << "Values in samples are: \n";
		for (int sample_number = 0; sample_number < _size; sample_number++) {
			for (int feature_number = 0; feature_number <= _depth; feature_number++) {
				cout << _inputs[sample_number][feature_number] << "\t";
			}
			if (_has_outputs) cout << ": " << _outputs[sample_number] << "\n";
		}
	}
	if (_is_normalized) {
		cout << "\nNormalized values: \n";
		for (int sample_number = 0; sample_number < _size; sample_number++) {
			for (int feature_number = 0; feature_number <= _depth; feature_number++) {
				cout << _normalized_inputs[sample_number][feature_number] << "\t";
			}
			if (_has_outputs) cout << ": " << _normalized_outputs[sample_number] << "\n";
		}
	}
	cout << "\nMaximum Inputs for each feature: ";
	for (int feature_number = 0; feature_number < _depth; feature_number++) {
		cout << _maximum_input[feature_number] << " ";
	}
	cout << "\nMinimum Inputs for each feature: ";
	for (int feature_number = 0; feature_number < _depth; feature_number++) {
		cout << _minimum_input[feature_number] << " ";
	}
	cout << "\nMaximum Output: " << _maximum_output << "\n";
	cout << "Minimum Output: " << _minimum_output << "\n";
}