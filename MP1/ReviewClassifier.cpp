#include "ReviewClassifier.h"
#include "stopwords.h"

bool replaceCharWithSpace(char c) {
	//return !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || c == ' ');
	return (c == ';'  || c == '.' || c == ',' || c == '-');
}

ReviewClassifier::ReviewClassifier() {
	totalWords = 0;
	numDocuments[POSITIVE] = 0;
	numDocuments[NEGATIVE] = 0;

	//create stopwords hashtable
	for (int i = 0; i < NUM_STOP_WORDS; i++) {
		stopWords[STOP_WORDS[i]] = 1;
	}
}

void ReviewClassifier::preprocess(string& review) {
	//lowercase everything
	std::transform(review.begin(), review.end(), review.begin(), ::tolower);

	//replace certain characers with spaces
	std::replace_if(review.begin(), review.end(), replaceCharWithSpace, ' ');
}

void ReviewClassifier::applyWordProbability(double& p, string& word, ReviewClass classification) {
	double a = 1.0;
	double am = a * (totalWords);
	//if word exists in this class
	if (wordCounts[classification].find(word) != wordCounts[classification].end()) {
		//and in the other
		if (wordCounts[!classification].find(word) != wordCounts[!classification].end()) {
			p += log((wordCounts[classification][word] + a) / (double)(wordCounts[POSITIVE][word] + wordCounts[NEGATIVE][word] + am));
		}
		else {
			p += log((wordCounts[classification][word] + a) / (double)(wordCounts[classification][word] + am));
		}
	}
	//otherwise if it exists only in the other class
	else if (wordCounts[!classification].find(word) != wordCounts[!classification].end()) {
		p += log(a / (double)(wordCounts[!classification][word] + am));
	}
}

double ReviewClassifier::probabilityOfClass(ReviewClass classification, string review) {
	preprocess(review);

	//words in this doc
	//only count a word once per doc
	unordered_map<string, int> docWords;

	//P(C | w1, w2, w3, ..., wn) = P(C)*P(w1 | C)*P(w2 |C)*...P(wn |C) * A
	//start with p at P(C)
	//double p = log((double)totalWords[classification] / (totalWords[ReviewClass::POSITIVE] + totalWords[ReviewClass::NEGATIVE]));
	double p = numDocuments[classification]/((double)numDocuments[POSITIVE] + numDocuments[NEGATIVE]);
	stringstream wss(review);
	string word, lastWord = "", lastLastWord = "", lastLastLastWord = "";
	while (wss >> word) {
		//skip stop words
		//if (stopWords.find(word) != stopWords.end())
		//	continue;
		//only count words once per doc
		//but still create bigrams
		if (docWords.find(word) == docWords.end()) {
			docWords[word] = 1;
			applyWordProbability(p, word, classification);
		}

		//create bi-gram with previous word
		if (lastWord != "") {
			string bigram = lastWord + " " + word;
			if (docWords.find(bigram) == docWords.end()) {
				docWords[bigram] = 1;
				applyWordProbability(p, bigram, classification);
			}

			//create tri-gram with previous two words
			if (lastLastWord != "") {
				string trigram = lastLastWord + " " + lastWord + " " + word;
				if (docWords.find(trigram) == docWords.end()) {
					docWords[trigram] = 1;
					applyWordProbability(p, trigram, classification);
				}

				if (lastLastLastWord != "") {
					string quadgram = lastLastLastWord + " " + lastLastWord + " " + lastWord + " " + word;
					if (docWords.find(quadgram) == docWords.end()) {
						docWords[quadgram] = 1;
						addWord(quadgram, classification);
					}
				}
			}
		}
		lastLastLastWord = lastLastWord;
		lastLastWord = lastWord;
		lastWord = word;
	}
	return p;
}

void ReviewClassifier::Train(string review, ReviewClass classification) {
	preprocess(review);

	//words in this doc
	//only count a word once per doc
	unordered_map<string, int> docWords;

	//add to word count for all words in this review
	stringstream wss(review);
	string word, lastWord = "", lastLastWord="", lastLastLastWord = "";
	while (wss >> word) {
		//skip stop words
		//if (stopWords.find(word) != stopWords.end())
		//	continue;
		//only count words once per doc
		//but still create bigrams
		if (docWords.find(word) == docWords.end()) {
			docWords[word] = 1;
			addWord(word, classification);
		}

		//create bi-gram with previous word
		if (lastWord != "") {
			string bigram = lastWord + " " + word;
			if (docWords.find(bigram) == docWords.end()) {
				docWords[bigram] = 1;
				addWord(bigram, classification);
			}

			//create tri-gram with previous two words
			if (lastLastWord != "") {
				string trigram = lastLastWord + " " + lastWord + " " + word;
				if (docWords.find(trigram) == docWords.end()) {
					docWords[trigram] = 1;
					addWord(trigram, classification);
				}

				if (lastLastLastWord != "") {
					string quadgram = lastLastLastWord + " " + lastLastWord + " " + lastWord + " " + word;
					if (docWords.find(quadgram) == docWords.end()) {
						docWords[quadgram] = 1;
						addWord(quadgram, classification);
					}
				}
			}
		}
		lastLastLastWord = lastLastWord;
		lastLastWord = lastWord;
		lastWord = word;
	}

	//num of documents
	numDocuments[classification]++;
}


void ReviewClassifier::addWord(string& word, ReviewClass classification) {
	//if word doesn't exist yet, add it
	if (wordCounts[classification].find(word) == wordCounts[classification].end()) {
		wordCounts[classification][word] = 1;
		totalWords++;
	}
	else {
		//add 1 to count for this word for the current classification
		wordCounts[classification][word] = wordCounts[classification][word] + 1;
	}
	//add to total number of words in class
	
}

ReviewClass ReviewClassifier::PredictClass(string review) {
	//if it is more likely to be a positive review than a negative review
	double pPos = probabilityOfClass(POSITIVE, review);
	double pNeg = probabilityOfClass(NEGATIVE, review);
	//cout << "pos: " << pPos << " neg: " << pNeg << endl;
	return (ReviewClass)(pPos > pNeg);
}

void ReviewClassifier::PrintMostImportantFeautures(int x) {
	unordered_map<string, double> probabilities;
	for (auto it = wordCounts[POSITIVE].begin(); it != wordCounts[POSITIVE].end(); ++it) {
		double pPos = 0;
		string word = it->first;
		applyWordProbability(pPos, word, POSITIVE);
		double pNeg = 0;
		applyWordProbability(pNeg, word, NEGATIVE);

		probabilities[word] = pPos - pNeg;
	}

	for (auto it = wordCounts[NEGATIVE].begin(); it != wordCounts[NEGATIVE].end(); ++it) {
		string word = it->first;
		//don't add twice
		if (probabilities.find(word) == probabilities.end()) {
			double pPos = 0;
			applyWordProbability(pPos, word, POSITIVE);
			double pNeg = 0;
			applyWordProbability(pNeg, word, NEGATIVE);

			probabilities[word] = pPos - pNeg;
		}
	}
	cout << endl << "Most Important Features:" << endl;
	cout << "Positive:" << endl;
	//sort based on positive first
	std::vector<DoublePair> v(probabilities.begin(), probabilities.end());
	std::partial_sort(v.begin(), v.begin() + x, v.end(), PairCompareGreater<DoublePair>());

	for (int i = 0; i < x; i++) {
		cout << i << ": " << v[i].first << " \t" << v[i].second << endl;
	}
	cout << endl <<  "Negative:" << endl;
	//then negative
	std::vector<DoublePair> v2(probabilities.begin(), probabilities.end());
	std::partial_sort(v2.begin(), v2.begin() + x, v2.end(), PairCompareLess<DoublePair>());

	for (int i = 0; i < x; ++i) {
		cout << i << ": " << v2[i].first << " \t" << v2[i].second << endl;
	}
}

void ReviewClassifier::RemoveAllBelow(int frequency) {
	unordered_map<string, int> freq;
	//count total frequencies of each feauture
	for (auto it = wordCounts[POSITIVE].begin(); it != wordCounts[POSITIVE].end(); ++it) {
		string word = it->first;
		freq[word] = it->second;
	}
	for (auto it = wordCounts[NEGATIVE].begin(); it != wordCounts[NEGATIVE].end(); ++it) {
		string word = it->first;
		if (freq.find(word) != freq.end())
			freq[word] = freq[word] + it->second;
		else
			freq[word] = it->second;
	}

	//sort by lowest to highest freq
	std::vector<IntPair> v(freq.begin(), freq.end());
	std::sort(v.begin(), v.end(), PairCompareLess<IntPair>());

	//remove rest from pos and neg
	for (int i = 0; i < v.size(); i++) {
		int count = v[i].second;
		if (count > frequency)
			break;
		if (wordCounts[POSITIVE].erase(v[i].first) == 1)
			totalWords -= 1;;
		if (wordCounts[NEGATIVE].erase(v[i].first) == 1)
			totalWords -= 1;;
	}
}

void ReviewClassifier::ReduceToMostFrequent(int amount) {
	unordered_map<string, int> freq;
	//count total frequencies of each feauture
	for (auto it = wordCounts[POSITIVE].begin(); it != wordCounts[POSITIVE].end(); ++it) {
		string word = it->first;
		freq[word] = it->second;
	}
	for (auto it = wordCounts[NEGATIVE].begin(); it != wordCounts[NEGATIVE].end(); ++it) {
		string word = it->first;
		if (freq.find(word) != freq.end())
			freq[word] = freq[word] + it->second;
		else
			freq[word] = it->second;
	}

	//sort by highest freq
	std::vector<IntPair> v(freq.begin(), freq.end());
	std::partial_sort(v.begin(), v.begin() + amount, v.end(), PairCompareGreater<IntPair>());

	/*cout << v[amount-1].second << endl;
	cout << v[amount-1].first << endl;
	cout << totalWords << endl;
	cout << v[amount-1].second/(double)totalWords << endl;*/

	//remove rest from pos and neg
	for (int i = amount; i < v.size(); i++) {
		int count = v[i].second;
		if (wordCounts[POSITIVE].erase(v[i].first) == 1)
			totalWords -= 1;;
		if (wordCounts[NEGATIVE].erase(v[i].first) == 1)
			totalWords -= 1;;
	}
}

double ReviewClassifier::TestOnFile(string file, bool print) {
	ifstream testFile(file);
	int numCorrect = 0;
	int totalCount = 0;
	string line;
	while (getline(testFile, line)) {
		stringstream ss(line);
		getline(ss, line, '\t');
		int classification;
		ss >> classification;
		ReviewClass predicted = PredictClass(line);
		if (predicted == classification)
			numCorrect++;
		totalCount++;

		if(print)
			cout << predicted << endl;
	}
	return (double)numCorrect / totalCount;
}