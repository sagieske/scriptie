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
	corpus_weights = {}

	totalPos = 0
	totalNeg = 0

	MIN_RANGE = -1
	MAX_RANGE = 1

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
		self.findHighest(self.corpus_weights, 10)
		self.findLowest(self.corpus_weights, 10)
		
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
		counter = 0
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
					self.corpus_weights[key] = math.copysign(math.pow(math.fabs(valueweight),2), valueweight)
				else:
					counter += 1
		self.scaleCorpusWeights()

	def scaleCorpusWeights(self):
		""" Scale weights of corpus to [MIN_RANGE, MAX_RANGE] """

		oldMax = max(self.corpus_weights.iteritems(), key=operator.itemgetter(1))[1]
		oldMin = min(self.corpus_weights.iteritems(), key=operator.itemgetter(1))[1]

		if(oldMin == oldMax):
			print "! No scaling, minimum is same as maximum (%d)" %oldMin
		else:
			print "scaling from [%f,%f] to [%f,%f]" %(float(oldMax),float(oldMin), self.MIN_RANGE, self.MAX_RANGE)
			# Set new range
			newMin = float(self.MIN_RANGE)
			newMax = float(self.MAX_RANGE)
		
			# Calculate new values
			delkeys = []
			for key in self.corpus_weights:
				oldValue = float(self.corpus_weights[key])
				newValue = (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
				if(newValue < 0.25 and newValue > -0.25):
					delkeys.append(key)
				else:	
					self.corpus_weights[key] = newValue

			for key in delkeys:
				del self.corpus_weights[key]

	def findHighest(self,corpus, nr):
		""" Print out max <nr> of corpus """
		print ">> %d max of corpus" %nr
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s, %s) : %f" % (item[0], item[1], value)

	def findLowest(self,corpus, nr):
		""" Print out min <nr> of corpus """
		print ">> %d min of corpus" %nr
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s, %s) : %f" % (item[0], item[1], value)


#b = BagOfWords([["test", "nee", "hello"],["nla", "wfr" , "gh", "sfe"], ["a", "asdf", "sdfsd"],["a", "asdf", "sdfsd"]], {0: 1, 1: 0, 2: 0, 3: 0}, [0,2,3])

