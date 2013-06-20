from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing					# used for scaler
from sklearn import metrics
import numpy as np
import helpers



class Start_SVM(object):
	""" Class for SVM classification
	"""

	#array = []
	classifier = None		# Classifier of object
	scaler = None			# Scaler for data
	train_vectors = []
	test_vectors = []
	train_tweetclasses = []
	test_tweetclasses  = []


	def __init__(self, pr_array, tweetclass, testmode, tuplebows, crossvalidation):
		""" Initialize items """
		self.pr_array = pr_array
		self.tweetclass = tweetclass
		self.testmode = testmode
		self.CROSS_VALIDATION = crossvalidation
		self.posbow, self.negbow = tuplebows
		

	def start_svm_testing(self, mode, minborder, maxborder, lenbow):
		""" Start classification training of SVM"""
		self.mode = mode

		self.scaler = None	# Reset scaling

		# Create data (classes, vectors)
		self.train_tweetclasses, self.train_vectors = self.svm_create_traintestdata(self.pr_array)


		# Run SVM		
		results = self.run_svm2(self.train_vectors, np.array(self.train_tweetclasses), self.CROSS_VALIDATION)

		return results

	def load_classifier(self, filename):
		""" Load classifier and scaler from file and set as class variables"""
		(classifier, scaler) = helpers.read_from_file(filename)
		self.classifier = classifier
		self.scaler = scaler
		
	def dump_classifier(self, filename):
		""" Dump classifier and scaler to file """
		dumptuple = (self.classifier, self.scaler)
		helpers.dump_to_file(filename, dumptuple)


	def start_classification(self, new_data):
		""" Start classification of twitter using classifier. New_data is array of tweets divided in tokens"""
		new_data_scaled = self.svm_create_vectorarray(new_data, self.scaler)	
		y_pred = self.classifier.predict(np.array(new_data_scaled))
		return y_pred
		
	def run_svm(self, X_train, y_train, X_test, y_test, k):
		""" Run SVM classifier. Configure parameters for SVM using grid search, then fit with cross valiation.
		Then predict test set. Return best parameters and scores.
		"""

		clf = svm.SVC()

		# Parameter grid
		param_grid = [
		 {'C': np.logspace(1,5,5), 'gamma': np.logspace(-3,0,5), 'kernel': ['rbf']}
		]

		score_func = metrics.f1_score
		clf = GridSearchCV(SVC(), param_grid, score_func=score_func,  n_jobs=-1 )

		###print "** Fitting SVM classifier.."
		clf.fit(X_train, y_train, cv=k)
		self.classifier = clf

		# Get best parameters
		dict_param = clf.best_params_
		gamma1 = dict_param['gamma']
		c = dict_param['C']

		# Get scores
		###print "** Run SVM classifier.."
		y_true, y_pred = y_test, clf.predict(X_test)
		tuples = metrics.precision_recall_fscore_support(y_true, y_pred)

		return (tuples, gamma1, c)

	def run_svm2(self, inputdata, outputdata, k):
		print "RUN SVM 2"
		# Cross validation
		cv = cross_validation.KFold(inputdata.shape[0], n_folds=k, indices=True,shuffle=True)

		cv_svm = []
		f1_svm = []
		for traincv, testcv in cv:
			clf_cv = svm.SVC()

			# Parameter grid
			param_grid = [
			 {'C': np.logspace(1,5,5), 'gamma': np.logspace(-3,0,5), 'kernel': ['rbf']}
			]

			score_func = metrics.f1_score
			clf_cv = GridSearchCV(SVC(), param_grid, score_func=score_func,  n_jobs=-1 )
			clf_cv.fit(inputdata[traincv], outputdata[traincv])
			test_classes_cv, y_pred_cv = outputdata[testcv], clf_cv.predict(inputdata[testcv])
			tuples = metrics.precision_recall_fscore_support(test_classes_cv, y_pred_cv)

			dict_param = clf_cv.best_params_
			gamma1 = dict_param['gamma']
			c = dict_param['C']

			f1 = metrics.f1_score(test_classes_cv, y_pred_cv)
			svm_tuple = (gamma1, c)
			cv_svm.append(svm_tuple)
			f1_svm.append(f1)

		
		# Get best classifier
		#(f1, gamma1, c) = max(cv_svm,key=operator.itemgetter(0))

		
		self.classifier = clf_cv
		print "score average: %s" + str(np.mean(f1_svm))
		print f1_svm

		return (tuples, gamma1, c)


	def svm_create_traintestdata(self, array):
		""" creates dataset needed for training/testing of SVM"""
		class_set = []
		tweetarray = []
		tweetarray = array
		class_set = self.tweetclass.values()

		X_scaled = self.svm_create_vectorarray(tweetarray, self.scaler)

		return (class_set, X_scaled)		

	def svm_create_vectorarray(self, array, scaler):
		""" Create vector array for classification, scale appropriate"""
		vecformat_array = []
		for tweet in array:
			# Different vectormodes
			if ('pn-neutral' in self.mode):
				vec = self.tweet_to_vector_posnegneutral(tweet, posbow, negbow)
			if ('posneg' in self.mode):
				totalbow = dict(self.posbow.items() + self.negbow.items())
				vec =  self.tweet_to_vector(tweet, totalbow, True)
			if ('pos1' in self.mode):
				vec =  self.tweet_to_vector(tweet, self.posbow, True)
			if ('neg1' in self.mode):
				vec =  self.tweet_to_vector(tweet, self.negbow, True)
			if ('freq' in self.mode):
				totalbow = dict(self.osbow.items() + self.negbow.items())
				vec =  self.tweet_to_vector(tweet, totalbow, False)

			vecformat_array.append(vec)
		X = np.array(vecformat_array)

		# Scale train and test data with same scaler
		if self.scaler is None:
			scaler = preprocessing.StandardScaler().fit(X)
			self.scaler = scaler


		X_scaled = self.scaler.transform(X)  

		return X_scaled

	def tweet_to_vector(self, tweet_token, totalbow, boolean):
		""" Convert tweet to binary vector for occurances in BOW"""
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		vec = []

		for index,x in enumerate(totalbow):
			if (' '.join(x) in tweetstring):
				if (boolean):
					vec.append(1.0)
				else:
					vec.append(totalbow[x])
			else:
				vec.append(0.0)

		return vec
			
				
	def tweet_to_vector_posnegneutral(self, tweet_token, posbow, negbow):
		""" Convert tweet to vector [posvalue, negvalue, neutralvalue]"""
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		posvalue = 0
		negvalue = 0
		neutralvalue = 0

		# Get all single words from bag of words
		poswords = helpers.unfold_tuples(posbow)	
		negwords = helpers.unfold_tuples(negbow)		
		totalwords = list(set(poswords + negwords))

		for x in posbow:
			if (' '.join(x) in tweetstring):
				posvalue += posbow[x]
		for x in negbow:
			if (' '.join(x) in tweetstring):
				negvalue += negbow[x]
		for word in tweet_token:
			if ( word not in totalwords ):		# neutral words
				neutralvalue += 1

		vec  = [float(posvalue), float(negvalue), float(neutralvalue)]
		return vec


