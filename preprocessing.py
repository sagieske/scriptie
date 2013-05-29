import nltk											# used for tokenization
import os											# used for calling Frog in new terminal
import operator
from pynlpl.clients.frogclient import FrogClient	# used for Frog client
import subprocess									# used for calling Frog in new terminal
import signal										# used for calling Frog in new terminal
import math	 
import pickle										# write and load list to file
import nltk											# used for stemmer
import re											# used for regex
import time											# used for sleep

class Preprocessing(object):
	"""
	Class for preprocessing tweets
	blablaballa etc

	"""
	# Debug files
	DEBUG_STEM = "debug_stem.txt"
	DEBUG_TOKEN = "debug_token.txt"
	DEBUG_LEMMA = "debug_lemma.txt"
	DEBUG_POS = "debug_pos.txt"

	PORTNUMBER = 1150	# PORTNUMBER for Frog

	tweets = {}
	stemmed_tweets_array = []
	tokenized_tweets_array = []
	lemmatized_tweets_array = []
	pos_tweets_array = []

	def __init__(self, mode, tweetlist):
		"""	Initialize tweets for use"""
		self.tweets = tweetlist
  		self.debug = "--debug" in mode
  		self.dump = "--write" in mode
		self.mode = re.sub(r' --(\S+)', r'', mode)


		#self.preprocess_tweets()

	def preprocess_tweets(self):
		"""	Call functions to process tweets according to mode"""
		print "** Preprocessing with MODE: %s\n   DEBUG: %s\n   WRITE TO FILE: %s" % (self.mode, self.debug, self.dump)
		mode_args = self.mode.split()
		if("stem" in mode_args):
			self.stemming()
			#self.stemming_str()				
		if("token" in mode_args):
			self.tokenize()
			#self.tokenize_str()
		if("frog" in mode_args):
			self.frogtokens()
			#if("lemma" in mode_args):
			#	self.lemmatized_str()
			#if("pos" in mode_args):
			#	self.pos_str()

		if ( self.dump ):
			self.write_all_to_file()
				
	def stemming_str(self):
		""" Print out stemmed_tweets_array as string"""
		print ">> Stemmed:"
		for item in self.stemmed_tweets_array:
			print '- ' + '|'.join(item)

	def tokenize_str(self):
		""" Print out tokenized_tweets_array as string"""
		print ">> Tokenized:"
		for item in self.tokenized_tweets_array:
			print '- ' + '|'.join(item)

	def lemmatized_str(self):
		""" Print out lemmatized_tweets_array as string"""
		print ">> Lemmatized:"
		for item in self.lemmatized_tweets_array:
			print '- ' + '|'.join(item)

	def pos_str(self):
		""" Print out pos_tweets_array as string"""
		print ">> POS tagged:"
		for item in self.pos_tweets_array:
			print '- ' + '|'.join(item)

	def stemming(self):
		""" Stem all tweets given to object and set to array """
		print "** Stemming.."
		debug = self.debug
		if (debug):
			try:
				self.read_from_file(self.DEBUG_STEM, "stem")
			except:
				print "! Error in reading from file debug.txt. Redo stemming"
				debug = False
		if (not debug):
			for index in self.tweets:
				# Snowball stemmer has problem with ascii, delete characters
				tweet_ascii = filter(lambda x: ord(x) < 128, self.tweets[index])
				stemmed_tweet = self.stem_tweet(tweet_ascii)		
				self.stemmed_tweets_array.append(stemmed_tweet)

	def stem_tweet(self, tweet):
		""" Stem tweet string and return array of stemmed words """
		stemmer = nltk.stem.SnowballStemmer('dutch')
		stemmed_tweet = stemmer.stem(tweet)
		stemmed_tweet = stemmed_tweet.split()
		return stemmed_tweet
		
	def tokenize(self):
		""" Tokenize all tweets given to object and set to array """
		print "** Create tokens.."
		debug = self.debug
		if (debug):
			try:
				self.read_from_file(self.DEBUG_TOKEN, "token")
			except:
				print "! Error in reading from file debug.txt. Redo tokenization"
				debug = False
		if (not debug):
			for index in self.tweets:
				tokens = nltk.word_tokenize(self.tweets[index])
				self.tokenized_tweets_array.append(tokens)

	def frogtokens(self):
		""" Lematize all tweets given to object using frog server/client and set to array """
		debug = self.debug
		if (debug):
			try:
				if( "lemma" in self.mode ):
					self.read_from_file(self.DEBUG_LEMMA, "lemma")
				if( "pos" in self.mode ):
					self.read_from_file(self.DEBUG_POS, "pos")
			except:
				print "! Error in reading from file debug.txt. Redo frogtokens"
				debug = False
		if(not debug):
			self.startFrogServer('start')			
			time.sleep(15)							# Time for startup server
			frogclient = FrogClient('localhost',self.PORTNUMBER)
			if( "lemma" in self.mode ):
				print "** Creating lemma's.."

			if ( "pos" in self.mode ):
				print "** Creating POS tags.."
			
			# Get frog analyses
			for index in self.tweets:
				tokensword, tokenslemma, tokenspos = self.frog_tweets(frogclient, self.tweets[index])
				if ( tokenslemma ):
					self.lemmatized_tweets_array.append(tokenslemma)
				if ( tokenspos ):
					self.pos_tweets_array.append(tokenspos)
			self.startFrogServer('stop')

	def startFrogServer(self, mode):
		""" Starts/stops Frog server in seperate terminal """
		if(mode == 'start'):
			print "** Start Frog Server"
			os.system("mate-terminal -e 'frog -S " + str(self.PORTNUMBER) + "'")
		if(mode == 'stop'):
			print "** Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)

	def frog_tweets(self, frogclient, tweet):
		"""	Use frog for processing according to mode. Return array of processed words """
		frogtweet = frogclient.process(tweet.lower())
		tokensword = []
		tokenslemma = []
		tokenspos = []
		for test in frogtweet:
			# Frog sometimes contains tuple of None
			if (None in test):
				pass
			else:
				word, lemma, morph, pos = test
				if('word' in self.mode):
					tokensword.append(word)
				if('lemma' in self.mode):
					tokenslemma.append(lemma)
				if('pos' in self.mode):
					tokenspos.append(pos)
		return (tokensword, tokenslemma, tokenspos)

	def read_from_file(self, filename, frogmode):
		"""	Load array from file """
		f = file(filename, "r")

		# Load into correct array
		if ( "stem" in frogmode):
			self.stemmed_tweets_array = pickle.load(f)
		if ( "token" in frogmode):
			self.tokenized_tweets_array = pickle.load(f)
		if ( "pos" in frogmode): 
			self.pos_tweets_array = pickle.load(f)
		if ( "lemma" in frogmode):
			self.lemmatized_tweets_array = pickle.load(f)

	def write_to_file(self, filename, array):
		"""	Dump array to file """
		f = file(filename, "w")
		pickle.dump(array, f)

	def write_all_to_file(self):
		"""	Dumps all filled arrays to file """
		if ( self.stemmed_tweets_array ):
			self.write_to_file(self.DEBUG_STEM, self.stemmed_tweets_array)
		if ( self.tokenized_tweets_array ):
			self.write_to_file(self.DEBUG_TOKEN, self.tokenized_tweets_array)
		if ( self.lemmatized_tweets_array ):
			self.write_to_file(self.DEBUG_LEMMA, self.lemmatized_tweets_array)
		if ( self.pos_tweets_array ):
			self.write_to_file(self.DEBUG_POS, self.pos_tweets_array)
