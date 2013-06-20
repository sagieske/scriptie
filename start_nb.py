from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn import metrics
import helpers
import numpy as np
import operator


class Start_NB(object):
	""" Class for Naive Bayes classification
	"""

	classifier = None		# Classifier of object
	transformer = None		# Transformer for data
	vectorizer = None		# Vectorizer for data
	
	traintweets = []
	train_classes = []
	testtweets = []
	test_classes = []


	def __init__(self, pr_array, mode, tweetclass, testmode, tuplebows, ngrams, crossvalidation):
		""" Initialize items """
		self.pr_array = pr_array
		self.tweetclass = tweetclass
		self.testmode = testmode
		self.CROSS_VALIDATION = crossvalidation
		self.posbow, self.negbow = tuplebows
		self.ngrams = ngrams
		self.mode = mode

	def start_classification(self, mode, new_data, allwords, fitclassifier):
		""" Start classification of twitter using classifier. New_data is array of tweets divided in tokens"""

		self.train_tweetclasses, self.train_vectors = self.nb_create_traintestdata(self.pr_array)

		if (not fitclassifier):
			self.classifier = MultinomialNB()
			self.classifier.fit(self.train_vectors, self.train_tweetclasses)
			if '--debug' in mode:
				self.dump_classifier("classifiertest_nb.txt")
		else:
			self.load_classifier("classifiertest_nb.txt")

		new_data_scaled = self.nb_create_inputdata(new_data, allwords)	

		y_pred = self.classifier.predict(new_data_scaled)
		return y_pred


	def start_naivebayes_evaluation(self, mode, minborder, maxborder, lenbow):
		""" Start classification training of Naive Bayes"""
		self.transformer = None	# Reset transformer
		self.vectorizer = None	# Reset transformer

		allwords = False
		if 'allwords' in self.mode:
			allwords = True

		self.train_tweetclasses, self.train_vectors = self.nb_create_traintestdata(self.pr_array, allwords=allwords)

		# Run Naive Bayes
		results = self.run_naivebayes_evaluation(self.train_vectors, np.array(self.train_tweetclasses), self.CROSS_VALIDATION)

		return results
	"""
	def start_classification(self, new_data):
		""" """Start classification of twitter using classifier. New_data is array of tweets divided in tokens""""""
		new_data_scaled = self.nb_create_vectorarray(new_data, self.scaler)	
		y_pred = self.classifier.predict(np.array(new_data_scaled))
		return y_pred
	"""


	def run_naivebayes_evaluation(self, inputdata, outputdata, k):
		""" Fit Naive Bayes Classification on train set with cross validation. 
		Run Naive Bayes Classificaiton on test set. Return results
		"""

		###print "** Fitting Naive Bayes classifier.."

		# Cross validation
		cv = cross_validation.KFold(inputdata.shape[0], n_folds=k, indices=True)
		cv_naivebayes = []
		f1_scores = []
		for traincv, testcv in cv:

			clf_cv = MultinomialNB()
			clf_cv.fit(inputdata[traincv], outputdata[traincv])

			y_pred_cv = clf_cv.predict(inputdata[testcv])

			f1 = metrics.f1_score(outputdata[testcv], y_pred_cv, pos_label=0)
			f1_scores.append(f1)

		
		#TODO: NEEDED? self.classifier = clf_cv
		print "score average: %s" + str(np.mean(f1_scores))

		average_score =np.mean(f1_scores)
		tuples = (average_score, f1_scores)

		return (tuples, 'N.A.', 'N.A.')

	def nb_create_traintestdata(self, array, **kwargs):
		""" Creates dataset needed for training/testing of Naive Bayes"""
		allwords = kwargs.get('allwords', False)
		classes = self.tweetclass.values()

		inputdata = self.nb_create_inputdata(array, allwords)

		return (classes, inputdata)


	def nb_create_inputdata(self, tweets, allwords):
		""" Create inputdata for Naive Bayes classifier. Return data
		"""

		# Select BOW type
		if ('posneg' in self.mode):
			bow = dict(self.posbow.items() + self.negbow.items())
		if ('pos1' in self.mode):
			bow =  self.posbow
		if ('neg1' in self.mode):
			bow = self.negbow
		inputdata = []
		if ( allwords ):
			for item in tweets:
				inputdata.append(' '.join(item))
		else:
			for item in tweets:
				bowtweet = self.get_bowtweet(item, bow)
				inputdata.append(bowtweet)

		# Convert collection to matrix of token counts
		if self.vectorizer is None:
			vectorizer = CountVectorizer(ngram_range=(self.ngrams[0],self.ngrams[len(self.ngrams)-1]))
			self.vectorizer = vectorizer.fit(inputdata)

		X_train_counts = self.vectorizer.transform(inputdata)

		# Transform count matrix to normalized tfidf representation
		self.transformer = None
		if self.transformer is None:
			tfidf_transformer = TfidfTransformer()
			self.transformer = tfidf_transformer.fit(X_train_counts)

		inputdata_fitted = self.transformer.transform(X_train_counts)


		return inputdata_fitted


	def get_bowtweet(self, tweet, bow):
		""" Get modified tweet with only words in bow"""
		tuple_array = self.splitbow(bow)
		
		listtweets = []
		# Set default to False
		for i in range(0, len(self.ngrams)):
			listtweet = [False]*len(tweet)
			listtweets.append(listtweet)
		
		# Get array of Booleans for words occuring in BOW
		for index_n, ngram in enumerate(self.ngrams):
			for index_t, word in enumerate(tweet):
				wordarray = []
				for item in tweet[index_t:index_t+ngram]:
					word = ''.join([x for x in item if ord(x) <128])		# Avoid problems with ascii values
					wordarray.append(word)
				wordstring = ' '.join(wordarray)
				if wordstring in tuple_array[index_n]:
					for i in range(index_t,index_t+ngram):
						listtweets[index_n][i] = True

		values = zip(*listtweets)

		# Create new tweet according to booleans
		new_tweet_array = []
		for index, word in enumerate(tweet):
			if ( any(values[index]) ):
				new_tweet_array.append(word)
		new_tweet = ' '.join(new_tweet_array)
		return new_tweet

	def splitbow(self, bow):
		""" Split BOW in arrays of tuples with same length """
		splitted_bowkeys = []
		for ngram in self.ngrams:
			ngram_array = []
			for item in bow:
				if len(item) == ngram:
					ngram_array.append(item)
			test = helpers.unfold_tuples_strings(ngram_array)
			splitted_bowkeys.append(test)

		return splitted_bowkeys

	def load_classifier(self, filename):
		""" Load classifier and scaler from file and set as class variables"""
		(classifier, transformer, vectorizer) = helpers.read_from_file(filename)
		self.classifier = classifier
		self.scaler = scaler
		
	def dump_classifier(self, filename):
		""" Dump classifier and scaler to file """
		dumptuple = (self.classifier, self.transformer, self.vectorizer)
		helpers.dump_to_file(filename, dumptuple)

