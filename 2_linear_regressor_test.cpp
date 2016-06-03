/*
	Simple program to test Dataset class
*/
#include "lib/linear_regressor.h"
#include<bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[])
{
	ifstream fin ("Datasets/sum_function_size_5_depth_2.txt");
	Dataset d;
	d.readDatasetFile(fin, 5, 2, true);
	// d.normalize();
	LinearRegressor lr;
	lr.train(&d, 0.01, 0.0001);
	lr.printResults();
	return 0;
}