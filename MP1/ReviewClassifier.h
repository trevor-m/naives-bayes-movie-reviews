#ifndef REVIEW_CLASSIFIER_H
#define REVIEW_CLASSIFIER_H

#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

enum ReviewClass {
	NEGATIVE = 0,
	POSITIVE = 1,
	COUNT = 2
};

//for preprocessing
bool replaceCharWithSpace(char c);

//a few templates to make sorting easier when determining 10 most important feautures
//and reducing to 10000 most frequent feautures
typedef std::pair<std::string, double> DoublePair;
typedef std::pair<std::string, double> IntPair;
template <typename T>
struct PairCompareLess {
	bool operator()(const T &lhs, const T &rhs) {
		return lhs.second < rhs.second;
	}
};
template <typename T>
struct PairCompareGreater {
	bool operator()(const T &lhs, const T &rhs) {
		return lhs.second > rhs.second;
	}
};


class ReviewClassifier {
private:
	//num of reviews with given class
	int numDocuments[ReviewClass::COUNT];
	//to keep track of total words(vocabulary)
	int totalWords;
	//to keep track of count of each word per class
	unordered_map<string, int> wordCounts[ReviewClass::COUNT];

	//hashtable of stopwords to remove
	unordered_map<string, int> stopWords;

	//preprocess a review before tokenizing
	//this is done when training and testing
	void preprocess(string& review);

	//CLASSIFICATION - these methods are used when classifying after the classifier has been trained
	//gets the P(word | C) and applies it to p
	void applyWordProbability(double& p, string& word, ReviewClass classification);
	//calculates P(C | review) by iterating over all words and applying their probabilities using applyWordProbability
	double probabilityOfClass(ReviewClass classification, string review);

	//TRAINING - these methods are used when training the classifier
	//adds to wordCounts
	void addWord(string& word, ReviewClass classification);

public:
	ReviewClassifier();
	
	//TRAINING - these methods are used when training the classifier
	void Train(string review, ReviewClass classification);
	void ReduceToMostFrequent(int amount);
	void RemoveAllBelow(int frequency);
	double TestOnFile(string file, bool print);

	//CLASSIFICATION - these methods are used when classifying after the classifier has been trained
	ReviewClass PredictClass(string review);

	//Stats
	void PrintMostImportantFeautures(int x);
};

#endif