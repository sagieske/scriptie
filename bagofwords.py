import operator
import math

class BagOfWords(object):
	"""
	Class for preprocessing tweets
	blablaballa etc

	"""
	bow = {}
	tokenarray = {}
	tweet_classes = {}
	corpus = {}
	bow = {}

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
		for itemindex in trainset:
			self.tokenarray[itemindex] = total_tokenarray[itemindex]
			self.tweet_classes[itemindex] = total_tweetclasses[itemindex]

	def create_corpus(self, ngramsize):
		""" Create training corpus """
		for index, key in enumerate(self.tokenarray):
			tokens = self.tokenarray[key]
			tweetclass = self.tweet_classes[key]
			self.add_tokens_to_corpus(tokens, tweetclass, ngramsize)
		self.setCorpusWeights()
		self.find_highest(self.bow, 10)
		self.find_lowest(self.bow, 10)
		
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
		""" Set weights for words. Remove singular occurances. """
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
					negative = value_neg/float(self.totalNeg)

				valueweight =(positive - negative)
				if (valueweight != 0):
					self.bow[key] = valueweight
					# TODO: problem with smoothing of weights?!
					#self.bow[key] = math.copysign(math.pow(math.fabs(valueweight),2), valueweight)

		self.scaleCorpusWeights()

	def scaleCorpusWeights(self):
		""" Scale weights of corpus to [MIN_RANGE, MAX_RANGE] """

		oldMax = float(max(self.bow.iteritems(), key=operator.itemgetter(1))[1])
		print oldMax
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

	# TODO: TUPLE PRINTING!
	def find_highest(self,corpus, nr):
		""" Print out max <nr> of corpus """
		print ">> %d max of corpus" %nr
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s) : %f" % (item[0], value)

	def find_lowest(self,corpus, nr):
		""" Print out min <nr> of corpus """
		print ">> %d min of corpus" %nr
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s) : %f" % (item[0], value)


#b = BagOfWords([["test", "nee", "hello"],["nla", "wfr" , "gh", "sfe"], ["a", "asdf", "sdfsd"],["a", "asdf", "sdfsd"]], {0: 1, 1: 0, 2: 0, 3: 0}, [0,2,3])

