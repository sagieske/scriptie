import nltk											# used for tokenization
import os											# used for calling Frog in new terminal
import operator
from pynlpl.clients.frogclient import FrogClient	# used for Frog client
import subprocess									# used for calling Frog in new terminal
import signal										# used for calling Frog in new terminal
import math	 
import pickle										# write and load list to file
import nltk											# used for stemmer
import re

class Preprocessing(object):
	"""
	Class for preprocessing tweets
	blablaballa etc

	"""
	portnumber = 1150	# Portnumber for Frog

	stemmed_tweets_array = []
	tokenized_tweets_array = []
	lemmatized_tweets_array = []

	def __init__(self, mode, tweetarray):
		"""	Initialize tweetarray for use"""
		self.tweetarray = tweetarray
  		self.debug = "--debug" in mode
		self.mode = re.sub(r' --(\S+)', r'', mode)


		print "** Preprocessing with MODE: %s, DEBUG: %s" % (self.mode, self.debug)
		self.preprocess_tweets()

	def preprocess_tweets(self):
		mode_args = self.mode.split()
		if(mode_args[0] == "stem"):
			self.stemming()
			self.stemming_str()
		if(mode_args[0] == "token"):
			self.tokenize()
			self.tokenize_str()
		if(mode_args[0] == "frog"):
			self.frogtokens(mode_args[1])
			self.frogtokens_str()
				
	def stemming_str(self):
		print "** Stemmed:\n"
		for item in self.stemmed_tweets_array:
			print '|'.join(item)

	def tokenize_str(self):
		print "** Tokenized: \n"
		for item in self.tokenized_tweets_array:
			print '|'.join(item)

	def frogtokens_str(self):
		print "** Lemmatized: \n"
		for item in self.lemmatized_tweets_array:
			print '|'.join(item)

	def stemming(self):
		""" Stem all tweets given to object and set to array """
		if (self.debug):
			try:
				self.read_from_file("debug.txt", "stem")
				self.stemming_str()
			except:
				print "! Error in reading from file debug.txt"

		else:
			for item in self.tweetarray:
				stemmed_tweet = self.stem_tweet(item)		
				self.stemmed_tweets_array.append(stemmed_tweet)

	def stem_tweet(self, tweet):
		""" Stem tweet string and return array of stemmed words """
		stemmer = nltk.stem.SnowballStemmer('dutch')
		stemmed_tweet = stemmer.stem(tweet)
		stemmed_tweet = stemmed_tweet.split()
		return stemmed_tweet
		
	def tokenize(self):
		""" Tokenize all tweets given to object and set to array """
		if (self.debug):
			try:
				self.read_from_file("debug.txt", "token")
				self.stemming_str()
			except:
				print "! Error in reading from file debug.txt"
		else:
			for item in self.tweetarray:
				tokens = nltk.word_tokenize(item)
				self.tokenized_tweets_array.append(tokens)

	def frogtokens(self, frogmode):
		""" Lematize all tweets given to object using frog server/client and set to array """
		if (self.debug):
			try:
				self.read_from_file("debug.txt", "lemma")
				self.stemming_str()
			except:
				print "! Error in reading from file debug.txt"
		else:
			self.startFrogServer('start')			
			time.sleep(15)							# Time for startup server
			frogclient = FrogClient('localhost',self.portnumber)

			for item in self.tweetarray:
				tokens = self.frog_tweets(frogclient, item, frogmode)
				# Write tokens to file for later testing
				self.lemmatized_tweets_array.append(tokens)

			self.startFrogServer('stop')

	def startFrogServer(self, mode):
		""" Starts/stops Frog server in seperate terminal """
		if(mode == 'start'):
			print "Start Frog Server"
			os.system("mate-terminal -e 'frog -S " + str(self.portnumber) + "'")
		if(mode == 'stop'):
			print "Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)

	def frog_tweets(self, frogclient, tweet, frogmode):
		"""	Use frog for processing according to mode. Return array of processed words """
		frogtweet = frogclient.process(tweet.lower())
		tokens = []
		for test in frogtweet:
			# Frog sometimes contains tuple of None
			if (None in test):
				pass
			else:
				word, lemma, morph, pos = test
				if(frogmode == 'word'):
					tokens.append(word)
				if(frogmode == 'lemma'):
						tokens.append(lemma)
				if(frogmode == 'pos'):
					tokens.append(pos)
		return tokens

	def read_from_file(self, filename, mode):
		"""	Load array from file """
		f = file(filename, "r")

		# Load into correct array
		if mode == "stem":
			self.stemmed_tweets_array = pickle.load(f)
		if mode == "token": 
			self.tokenized_tweets_array = pickle.load(f)
		if mode == "lemma": 
			self.lemmatized_tweets_array = pickle.load(f)

	def write_to_file(self, filename, array):
		"""	Dump array to file """
		f = file(filename, "w")
		pickle.dump(array, f)

