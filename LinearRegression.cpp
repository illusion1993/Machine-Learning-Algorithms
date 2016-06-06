/*
	Simple program to test Dataset class
*/
#include "lib/linear_regressor.h"
#include<bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[])
{
	Dataset d, t, e, u;
	LinearRegressor lr, tr;


	ifstream fin ("Datasets/sum_function_size_5_depth_2.txt");
	d.readDatasetFile(fin, 5, 2, true);
	
	ifstream tin ("Datasets/sum_function_size_5_depth_2_test.txt");
	t.readDatasetFile(tin, 5, 2, false);

	/*
	Below dataset contains a linear equation in 4 variables (hyperplane_equation_size_50_depth_3.txt)
	For a linear equation of n variables, minimum of (n + 1) samples are required to find a solution
	So, for a linear equation of 1 variable like mx + c = y(eqn of a line), minimum 2 samples are required
	Which is correct because to find a line, you need at least 2 points that lie on it
	To find a plane, you need at least 3 points and for a hyperplane, 4 points and so on
	*/
	ifstream gin ("Datasets/hyperplane_equation_size_50_depth_3.txt");
	e.readDatasetFile(gin, 50, 3, true);
	
	ifstream uin ("Datasets/hyperplane_equation_size_50_depth_3_test.txt");
	u.readDatasetFile(uin, 50, 3, false);

	

	// Training and testing on simple sum function
	lr.train(&d, 0.02, 0.0000001);
	lr.printResults();
	cout << "\n\nNOW TESTING:\n\n";
	lr.test(&t);

	cout << "\n\nNEW PROBLEM---------------------\n";

	tr.train(&e, 0.005, 0.0000001);
	tr.printResults();
	cout << "\n\nNOW TESTING:\n\n";
	tr.test(&u);
	

	return 0;
}