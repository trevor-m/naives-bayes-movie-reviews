#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <iomanip>
#include "ReviewClassifier.h"

double trainOnSegment(ifstream& trainingFile, int count) {
	static int segnum = 0;
	ReviewClassifier classifier;
	int i = 0;
	string line;
	while (getline(trainingFile, line) && i < count) {
		stringstream ss(line);
		getline(ss, line, '\t');
		int classification;
		ss >> classification;
		classifier.Train(line, (ReviewClass)classification);
		i++;
	}
	classifier.ReduceToMostFrequent(30000);

	//test
	//test on testing file
	ifstream testFile("testing.txt");
	int numCorrect = 0;
	int totalCount = 0;
	while (getline(testFile, line)) {
		stringstream ss(line);
		getline(ss, line, '\t');
		int classification;
		ss >> classification;
		ReviewClass predicted = classifier.PredictClass(line);
		if (predicted == classification)
			numCorrect++;
		totalCount++;

		//cout << predicted << endl;
	}
	double testingAccuracy = (double)numCorrect / totalCount;
	cout << "segment " << segnum++ << ": " << testingAccuracy << endl;
	return testingAccuracy;
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		cout << "invalid or missing arguments" << endl;
		cout << "usage: ./NaiveBayesClassifier training.txt testing.txt" << endl;
	}


	int numReviews = 5000;
	int numSegs = 5;
	ifstream trainingFile(argv[1]);
	double avg = 0;
	for (int i = 0; i < numSegs; i++) {
		avg += trainOnSegment(trainingFile, numReviews/numSegs);
	}
	cout << "avg: " << avg / numSegs << endl;
	cin.get();

	/*time_t startTime, endTime;
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
	//classifier.RemoveAllBelow(8);
	classifier.ReduceToMostFrequent(20000);
	//stop timer
	time(&endTime);
	trainingTime = (int)difftime(endTime, startTime);

	//testing
	//test on training file
	//move back to beginning of training file
	trainingFile.clear();
	trainingFile.seekg(0, ios::beg);
	int numCorrect = 0;
	int totalCount = 0;
	while (getline(trainingFile, line)) {
		stringstream ss(line);
		getline(ss, line, '\t');
		int classification;
		ss >> classification;
		ReviewClass predicted = classifier.PredictClass(line);
		if (predicted == classification)
			numCorrect++;
		totalCount++;
	}
	trainingAccuracy = (double)numCorrect / totalCount;

	//test on testing file
	time(&startTime);
	ifstream testFile(argv[2]);
	numCorrect = 0;
	totalCount = 0;
	while (getline(testFile, line)) {
		stringstream ss(line);
		getline(ss, line, '\t');
		int classification;
		ss >> classification;
		ReviewClass predicted = classifier.PredictClass(line);
		if (predicted == classification)
			numCorrect++;
		totalCount++;

		cout << predicted << endl;
	}
	testingAccuracy = (double)numCorrect / totalCount;
	time(&endTime);
	testingTime = (int)difftime(endTime, startTime);

	//report results
	//report times
	cout << trainingTime << " seconds (training)" << endl;
	cout << testingTime << " seconds (labeling)" << endl;
	//report accuracy
	cout << fixed << setprecision(3) << trainingAccuracy << " (training)" << endl;
	cout << fixed << setprecision(3) << testingAccuracy << " (testing)"; //no endl after last

	//classifier.PrintMostImportantFeautures(10);*/
	return 0;
}