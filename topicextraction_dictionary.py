import csv
import gensim
import nltk
import re
import operator
import math
import helpers

class TopicExtraction_dictionary(object):

	COLUMN_TWEET = 3
	COLUMN_CLASS = 5

	corpusfile_tweets = []
	referencefile_tweets = []

	corpus1 = {}
	corpus2 = {}

	loglikelihood1 = {}
	loglikelihood2 = {}

	EMOTICONS = ['[=:]-*[()DdPpSsOo(\|)(\$)]']
	EMOTICONS_2 = ['\(.\)']
	NUMBERS = ['\d+[(\.):]\d+', '\d+']



	def __init__(self, mode, corpusfile, referencefile):
		""" Initialize tweets from files and dictionaries"""
		self.load_stopword_file()
		if '--debug' in mode:
			self.corpusfile_tweets = helpers.read_from_file("corpusfile_lda_testing.txt")
			self.referencefile_tweets = helpers.read_from_file("referencefile_lda_testing.txt")
		else:
			self.corpusfile_tweets = self.get_tweets(corpusfile)
			helpers.dump_to_file("corpusfile_lda_testing.txt", self.corpusfile_tweets)
			self.referencefile_tweets = self.get_tweets(referencefile)
			helpers.dump_to_file("referencefile_lda_testing.txt", self.referencefile_tweets)

		self.corpus = self.create_dictionary(self.corpusfile_tweets)
		self.referencecorpus = self.create_dictionary(self.referencefile_tweets)

		self.loglikelihood = self.calculate_loglikelihood(self.corpus, self.referencecorpus)


	def calculate_loglikelihood(self, corpus, referencecorpus):
		""" Calculate loglikelihood for all words in corpus"""
		loglikelihood_dict = {}
	
		totalwords_corpus = sum(corpus.values())
		totalwords_referencecorpus = sum(referencecorpus.values())

		#totalwords_corpus = sum(corpus.values())

		for word in corpus:
		
			word_corpus = float(corpus.get(word))
			word_referencecorpus = float(referencecorpus.get(word, 0))

			word_occurance = float(word_corpus + word_referencecorpus)
			total = float(totalwords_corpus + totalwords_referencecorpus)
			e1 = float(totalwords_corpus) * word_occurance/total
			e2 = float(totalwords_referencecorpus) * word_occurance/total
			"""
			try:
				part1 = word_corpus * math.log(word_corpus/e1)
				part2 = word_referencecorpus * math.log((word_referencecorpus/e2))
				loglikelihood_dict[word] = 2*(part1 + part2)
			except:
				loglikelihood_dict[word] = 0.0
			"""
			part1 = word_corpus * math.log(word_corpus/e1)
			try:
				part2 = word_referencecorpus * math.log((word_referencecorpus/e2))
			except:
				part2 = 0
			loglikelihood_dict[word] = 2*(part1 + part2)

			"""
			if word == 'werken':
				print "---"
				print word_corpus
				print totalwords_corpus
				print word_referencecorpus
				print totalwords_referencecorpus
				print word, loglikelihood_dict[word]
			"""

		return loglikelihood_dict

	def get_top_likelihood(self, dictionary, nr):
		""" Get top of loglikelihood"""
		topWords = dict(sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])

		return topWords

	def get_lowest_likelihood(self, dictionary, nr):
		""" Get top of loglikelihood"""
		lowestWords = dict(sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])

		return lowestWords

	def get_tweets(self, filename):
		""" Get tweets from file """
		print "Read tweets from CSV.."
		tweetarray = []
		DELIMITER = "\t"
		data = csv.reader(open(filename, 'rU'), delimiter=DELIMITER)
		for i, row in enumerate(data):
			if (row[self.COLUMN_CLASS] == '0'):
				stemmed_tweet = self.stem_tweet(row[self.COLUMN_TWEET])
				cleaned_tweet = self.remove_stopwords(stemmed_tweet.lower())
				tweetarray.append(cleaned_tweet)
		return tweetarray

	def create_dictionary(self, tweetarray):
		""" Create dictionary with frequency from array """
		print "Create dictionary"
		corpus = {}
		for tweet in tweetarray:
			tokens = nltk.word_tokenize(tweet)
			for token in tokens:
				corpus[token] = corpus.get(token,0) +1
		return corpus



	def stem_tweet(self, tweet):
		""" Return stemmed tweets"""
		stemmer = nltk.stem.SnowballStemmer('dutch')
		# Snowball stemmer has problem with c, delete characters
		tweet_ascii = filter(lambda x: ord(x) < 128, tweet)
		stemmed_tweet = stemmer.stem(tweet_ascii)
		return stemmed_tweet

	def load_stopword_file(self):
		stopword_file = open( "dutch-stop-words.txt", "r" )
		array = []
		for line in stopword_file:
			word = re.sub('\n','', line)
			array.append( word )
		self.STOPWORD_FILE = array
		
	def remove_stopwords(self,inputsentence):
		""" Removes stop words in sentences. Returns substituted sentence """

		wordlist = []
		#pattern_time = re.compile('vanavond|morgen|vandaag|vanmiddag|gister|gisteren|eerst|daarna')
		#sentence = pattern_time.sub('', sentence)
		

		# Delete 'links'
		r_link1 = re.compile(r"(http:)[^ ]*")
		sentence = r_link1.sub(' ', inputsentence)
		r_link2 = re.compile(r"(www.)[^ ]*")
		sentence = r_link2.sub(' ', sentence)

		# append words like z'n 
		sentence = re.sub("'", "", sentence)
		sentence = re.sub("[^\w\s]", " ", sentence)

		#self.STOPWORD_FILE +
		wordlist +=  self.NUMBERS + self.EMOTICONS + self.EMOTICONS_2
		for x in wordlist:
			sentence = re.sub(' '+x+' ',' ', sentence)
			sentence = re.sub('\A'+x+' ',' ', sentence)
			sentence = re.sub(' '+x+'\Z',' ', sentence)

		# Delete 'laughs'
		sentence = re.sub('ha(ha)+',' ', sentence)


		return sentence

#te = TopicExtraction_dictionary("--","day_output_day2_class.csv", "day_output_class.csv")
#te.log_likelihood1 = te.calculate_loglikelihood(te.corpus1, te.corpus2)
#te.log_likelihood2 = te.calculate_loglikelihood(te.corpus2, te.corpus1)
#print te.get_top_likelihood(te.loglikelihood,100)
#print te.get_lowest_likelihood(te.loglikelihood,100)
