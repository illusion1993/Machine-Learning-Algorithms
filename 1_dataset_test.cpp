#include "lib/linear_regression.h"
#include<bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[])
{
	ifstream fin ("Datasets/sum_function_size_5_depth_2.txt");
	Dataset d;
	d.readDatasetFile(fin, 5, 2, true);
	d.printDataset();
	return 0;
}