from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn import metrics
import helpers
import numpy as np
import operator


class Start_NB(object):
	"""blablbla
	"""

	#array = []
	classifier = None		# Classifier of object
	transformer = None			# Transformer for data
	traintweets = []
	train_classes = []
	testtweets = []
	test_classes = []


	def __init__(self, pr_array, tweetclass, trainset, testset, testmode, tuplebows, ngrams, crossvalidation):
		""" Initialize items """
		self.pr_array = pr_array
		self.tweetclass = tweetclass
		self.testmode = testmode
		self.trainset = trainset
		self.testset = testset
		self.CROSS_VALIDATION = crossvalidation
		self.posbow, self.negbow = tuplebows
		self.ngrams = ngrams

	def start_naivebayes_testing(self, mode, minborder, maxborder, lenbow):
		""" Start classification training of Naive Bayes"""
		self.mode = mode
		self.transformer = None	# Reset scaling

		allwords = False
		if 'allwords' in self.mode:
			allwords = True

		self.train_tweetclasses, self.train_vectors = self.nb_create_traintestdata(self.pr_array, self.trainset, allwords=allwords)
		self.test_tweetclasses, self.test_vectors = self.nb_create_traintestdata(self.pr_array, self.testset, allwords=allwords)

		# Run Naive Bayes
		results = self.run_naivebayes(np.array(self.train_vectors), np.array(self.train_tweetclasses), np.array(self.test_vectors), np.array(self.test_tweetclasses), self.CROSS_VALIDATION)

		return results


	def run_naivebayes(self, X_train, train_classes, X_test, test_classes, k):
		""" Fit Naive Bayes Classification on train set with cross validation. 
		Run Naive Bayes Classificaiton on test set. Return results
		"""

		# Transform train and test data
		count_vect = CountVectorizer(ngram_range=(self.ngrams[0],self.ngrams[len(self.ngrams)-1]))

		X_train_counts = count_vect.fit_transform(X_train)
		self.transformer = TfidfTransformer()
		X_train_tfidf = self.transformer.fit_transform(X_train_counts)

		X_new_counts = count_vect.transform(X_test)
		X_new_tfidf = self.transformer.transform(X_new_counts)

		###print "** Fitting Naive Bayes classifier.."

		# Apply cross validation
		cv = cross_validation.KFold(X_train_tfidf.shape[0], n_folds=k, indices=True)

		cv_naivebayes = []
		for traincv, testcv in cv:
			clf_cv = MultinomialNB()
			clf_cv.fit(X_train_tfidf[traincv], train_classes[traincv])
			test_classes_cv, y_pred_cv = train_classes[testcv], clf_cv.predict(X_train_tfidf[testcv])
			nb_tuple = (metrics.f1_score(test_classes_cv, y_pred_cv), clf_cv)
			cv_naivebayes.append(nb_tuple)
		
		# Get best classifier
		(f1, best_clf) = max(cv_naivebayes,key=operator.itemgetter(0))
		

		###print "** Run Naive Bayes classifier.."
		y_true, y_pred = test_classes, best_clf.predict(X_new_tfidf)

		tuples = metrics.precision_recall_fscore_support(y_true, y_pred)
		return (tuples, 'N.A.', 'N.A.')

	def nb_create_traintestdata(self, array, indexset, **kwargs):
		""""  """	
		allwords = kwargs.get('allwords', False)
		tweets = []
		classes = []

		if ( allwords ):
			for index in indexset:
				tweets.append(array[index])
				classes.append(self.tweetclass[index])
		else:
			for index in indexset:
				tweets.append(array[index])
				classes.append(self.tweetclass[index])

		inputdata = self.nb_create_inputdata(tweets, allwords)

		return (classes, inputdata)


	def nb_create_inputdata(self, tweets, allwords):
		""" """

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
			
		
		"""
		# Use all words in tweets
		if ( allwords ):
			for index in indexset:
				tweets.append(' '.join(self.pr_array[index]))
				classes.append(self.tweetclass[index])



		
		# Use words occuring in BOW of tweet
		else:
			for index in indexset:
				tweet = array[index]
				bowtweet = self.get_bowtweet(tweet, bow)
				tweets.append(bowtweet)
				classes.append(self.tweetclass[index])

		return (classes, tweets)
		"""

		return inputdata


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
