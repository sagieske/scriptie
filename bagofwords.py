import operator
import math

class BagOfWords(object):
	"""
	Class for creating and calculating Bag Of Words used for SVM classifier. Tokens are added to a corpus dictionary with their frequencies.
	A dictionary with weights for tokens can be created. A partial Bag of Words can be created, getting the specified top tokens indicating the 	positive class (activity) and top tokens indicating the negative class (non-activity) with scores higher/lower than set borders.

	Parameters:
	----------
	bow : dictionary with tuples with strings (n-gram length) as key and weights (float) as values.
		Dictionary for positive/negative impact of tokens in tuple.
		Positive value means positive impact on activity class. Negative value means negative impact on activity class. 
	corpus:

	tokenarray: 
	tweet_classes:

	totalPos: int
		Total tokens in positive tweets (activity class)

	totalNeg: int
		Total tokens in negative tweets (non-activity class)

	#TODO: next 4 are used for scaling in range. Is not used anymore??!
	MIN_RANGE: constant int
		Used for minimum border in scaling in range [MIN_RANGE, MAX_RANGE]

	MAX_RANGE: constant int
		Used for maximum border in scaling in range [MIN_RANGE, MAX_RANGE]

	POS_BORDER: constant int
		Used for minimum positive border for deletion in scaling in range [MIN_RANGE, MAX_RANGE]

	NEG_BORDER: constant int
		Used for minimum negative border for deletion in scaling in range [MIN_RANGE, MAX_RANGE]

	"""

	bow = {}
	corpus = {}
	tokenarray = {}
	tweet_classes = {}

	totalPos = 0
	totalNeg = 0

	# Ranges for scaling
	MIN_RANGE = -1
	MAX_RANGE = 1

	# Borderweights for bow
	POS_BORDER = 0.1
	NEG_BORDER = -0.1


	def __init__(self, total_tokenarray, total_tweetclasses, trainset):
		""" Initialize arrays according to trainset"""
		#self.bow = {}
		#self.tokenarray = {}
		#self.tweet_classes = {}
		#self.corpus = {}
		#self.bow = {}
		for itemindex in trainset:
			self.tokenarray[itemindex] = total_tokenarray[itemindex]
			self.tweet_classes[itemindex] = total_tweetclasses[itemindex]

	def create_corpus(self, ngramsize):
		""" Create training corpus """
		self.corpus = {}
		self.bow = {}
		for index, key in enumerate(self.tokenarray):
			tokens = self.tokenarray[key]
			tweetclass = self.tweet_classes[key]
			self.add_tokens_to_corpus(tokens, tweetclass, ngramsize)
		self.setCorpusWeights()
		
	def add_tokens_to_corpus(self,tokens,tweetclass, ngramsize):
		""" add every token to corpus accoding to class"""
		if(tweetclass == 0):
			addition = (1,0)
		else:
			addition = (0,1)

		for index in range(0, len(tokens)-ngramsize+1):
			# Create tuple
			tupleItem = (tokens[index],)
			for i in range(index+1,index+ngramsize):
				tupleItem = tupleItem + (tokens[i],)
	
			if ( addition == (1,0) ):
				self.totalPos +=1
			else:
				self.totalNeg +=1
			# Add ngrams in dictionary with addition
			if tupleItem in self.corpus:
				self.corpus[tupleItem] = tuple(map(operator.add, self.corpus[tupleItem], (addition)))
			else:
	 			self.corpus[tupleItem] = addition

	def setCorpusWeights(self):
		""" Set weights for words. Remove singular occurances. 
		Calculation for impact is dependent on occurance in pos/neg tweet
		"""
		for key,value in self.corpus.iteritems():
			value_pos, value_neg = value

			# Only use tokens that occur more than once
			if (sum(value) >1):				
				# Calculate positive and negative influence
				positive = 0
				negative = 0
				if (value_pos > 0):
					positive = value_pos/float(self.totalPos)
				if (value_neg > 0):
					negative = value_neg/ float(self.totalNeg)

				valueweight =(positive - negative)
				if (valueweight != 0):
					self.bow[key] = valueweight

		#self.scaleCorpusWeights()

	# TODO: Scaling is not done anymore here ?!
	def scaleCorpusWeights(self):
		""" Scale weights of corpus to [MIN_RANGE, MAX_RANGE] """

		oldMax = float(max(self.bow.iteritems(), key=operator.itemgetter(1))[1])
		oldMin = float(min(self.bow.iteritems(), key=operator.itemgetter(1))[1])

		if(oldMin == oldMax):
			print "! No scaling, minimum is same as maximum (%d)" %oldMin
		else:
			print "scaling from [%f,%f] to [%f,%f]" %(float(oldMin),float(oldMax), self.MIN_RANGE, self.MAX_RANGE)
			# Set new range
			newMin = float(self.MIN_RANGE)
			newMax = float(self.MAX_RANGE)
		
			delkeys = []
			for key in self.bow:
				oldValue = float(self.bow[key])
				newValue = (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
				if(newValue < self.POS_BORDER and newValue > self.NEG_BORDER):
					delkeys.append(key)
				else:
					self.bow[key] = newValue

			# Deletion of keys
			for key in delkeys:
				del self.bow[key]

	def bow_partial(self, **kwargs):
		""" Select part of Bag of Words and return dict """
		partial = {}
		nr = kwargs.get('nr',len(self.bow))
		min_border = kwargs.get('min_border',0)
		max_border = kwargs.get('max_border',0)
			
		positive = dict(sorted(self.bow.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])
		for key in positive:
			if ( positive[key] > max_border):
				partial[key] = positive[key]

		negative = dict(sorted(self.bow.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])
		for key in negative:
			if ( negative[key] < min_border):
				partial[key] = negative[key]

		return partial

	def find_lowest(self,corpus, nr):
		""" Print out <nr> of corpus with lowest values"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])

	def find_highest(self,corpus, nr):
		""" Get <nr> of corpus with highest values"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])

	def print_topcorpus(self,corpus,nr):
		""" Print out corpus """
		print ">> %d of corpus" %nr
		for item in corpus:
			value = corpus[item]
			print "(%s) : %f" % (','.join(item), value)




