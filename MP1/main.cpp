#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <iomanip>
#include "ReviewClassifier.h"

int main(int argc, char* argv[]) {
	if (argc != 3 || argc != 4) {
		cout << "invalid or missing arguments" << endl;
		cout << "usage: ./NaiveBayesClassifier training.txt testing.txt" << endl;
		cout << "usage: ./NaiveBayesClassifier training.txt testing.txt validation.txt" << endl;
	}
	time_t startTime, endTime;
	int trainingTime = 0;
	int testingTime = 0;
	double trainingAccuracy = 0.0;
	double testingAccuracy = 0.0;

	ReviewClassifier classifier;

	//training
	//start timer
	time(&startTime);
	//read file
	ifstream trainingFile(argv[1]);
	string line;
	while (getline(trainingFile, line)) {
		stringstream ss(line);
		getline(ss, line, '\t');
		int classification;
		ss >> classification;
		classifier.Train(line, (ReviewClass)classification);
	}

	//parameter optimization using validation set (if provided)
	if (argc == 4) {
		int reduceTo = 80000;
		int bestReduceTo = 80000;
		double bestAccuracy = 0.0;
		//copy original classifier
		ReviewClassifier optimizationClassifier = classifier;
		while (reduceTo >= 30000) {
			optimizationClassifier.ReduceToMostFrequent(reduceTo);
			//validate
			double acc = optimizationClassifier.TestOnFile(argv[3], false);
			if (acc > bestAccuracy) {
				bestReduceTo = reduceTo;
				bestAccuracy = acc;
			}
			reduceTo -= 2000;
		}
		//use param which gave best accuracy
		classifier.ReduceToMostFrequent(bestReduceTo);
	}
	else {
		//35000 most frequent features are kept
		classifier.ReduceToMostFrequent(35000);
	}

	//stop timer
	time(&endTime);
	trainingTime = (int)difftime(endTime, startTime);

	//testing
	//test on training file
	//move back to beginning of training file
	trainingFile.close();
	trainingAccuracy = classifier.TestOnFile(argv[1], false);

	//test on testing file
	time(&startTime);
	testingAccuracy = classifier.TestOnFile(argv[2], true);
	time(&endTime);
	testingTime = (int)difftime(endTime, startTime);

	//report results
	//report times
	cout << trainingTime << " seconds (training)" << endl;
	cout << testingTime << " seconds (labeling)" << endl;
	//report accuracy
	cout << fixed << setprecision(3) << trainingAccuracy << " (training)" << endl;
	cout << fixed << setprecision(3) << testingAccuracy << " (testing)"; //no endl after last

	//classifier.PrintMostImportantFeautures(10);
	return 0;
}