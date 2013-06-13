import csv											# used for import csv tweets
import os											# used for calling Frog in new terminal
import operator
import time											# used for timer
from pynlpl.clients.frogclient import FrogClient	# used for Frog client
import helpers
import pickle
import re

import logging, gensim, bz2

class TopicExtraction(object):
	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated.csv", 'rU'), delimiter=DELIMITER)
	class_dict = {"Y": 0, "N": 1, "U": 1}

	short_tweets = []
	tweets = []
	tokens_tweets_array = []
	pos_tweets_array = []

	PORTNUMBER = 1161

	def __init__(self):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		for i, row in enumerate(self.data):
			# Ignores header
			if(i == 0):
				pass
			else:
			# TEMP, for testing only
				begin = 1
				end = 1000
				if (i >= begin and i <= end):
					# TEMP, Only add if class is known! for testing only
					if (self.class_dict.get(row[5].upper()) is 0):
						self.tweets.append(row[3])
						substituted = re.sub(r'\.\.?', r'', row[3])
						short =re.split('\sen\s|\.|\,|\?|!| ;',substituted)
						print short
						print len(short)
						listindex = [item for item in range(len(short)) if 'vanavond' in short[item].lower()]
						for item in listindex:
							shortened_tweet = short[item]
							self.short_tweets.append(shortened_tweet)

	def tryout(self, array):
		self.startFrogServer('start')			
		time.sleep(15)							# Time for startup server
		frogclient = FrogClient('localhost',self.PORTNUMBER)

		print "** Creating POS tags.."
		for item in array:
			tokensword, tokenslemma, tokenspos = self.frog_tweets(frogclient, item)
			if ( tokensword ):
				self.tokens_tweets_array.append(tokensword)
			else:
				self.tokens_tweets_array.append(None)
			if ( tokenspos ):
				self.pos_tweets_array.append(tokenspos)
			else:
				self.pos_tweets_array.append(None)
		
		helpers.dump_to_file('te-token.txt', self.tokens_tweets_array)
		helpers.dump_to_file('te-pos.txt', self.pos_tweets_array)


	def startFrogServer(self, mode):
		""" Starts/stops Frog server in seperate terminal """
		if(mode == 'start'):
			print "** Start Frog Server"
			os.system("mate-terminal -e 'frog -S " + str(self.PORTNUMBER) + " > /dev/null 2>&1'")
		if(mode == 'stop'):
			print "** Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)
	
	def read_from_file(self):
		"""	Load array from file """
		fT = file('te-token.txt', "r")
		fP = file('te-pos.txt', "r")
		self.tokens_tweets_array = pickle.load(fT)
		self.pos_tweets_array = pickle.load(fP)

	def get_postags(self):
		for index, item in enumerate(self.short_tweets):
			token_tweet = self.tokens_tweets_array[index]
			pos_tweet = self.pos_tweets_array[index]
			print self.short_tweets[index]
			print pos_tweet
			string = ""
			for index, item in enumerate(pos_tweet):
				if 'WW' in str(item):
					string += str(token_tweet[index]) + " "
			print string
			print "--"

	def testout(self):
		flat = [item for sublist in self.pos_tweets_array for item in sublist]
		test = set(flat)
		dictionary_pos = {}
		for indextweet, item in enumerate(self.tokens_tweets_array):
			for index, pos in enumerate(self.pos_tweets_array[indextweet]):
				#position = [pos, self.pos_tweets_array[indextweet][index+1]]
				#print position
				dictionary_pos.setdefault(pos,[]).append(item[index].encode("utf8"))

		#for item in dictionary_pos:
		#	print str(item) + " " + str(dictionary_pos[item])

		c = csv.writer(open("postags.csv", "wb"),delimiter = ',')
		headings = (["POSTAG","VALUES"])
		c.writerow(headings)

		sorted_pos = dict(sorted(dictionary_pos.iteritems(), key=operator.itemgetter(1)))
		print sorted_pos
		for key in sorted_pos:
			row = []
			row.append(key)
			row += set(sorted_pos[key])
			c.writerow(row)


			

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
				tokensword.append(word)
				tokenslemma.append(lemma)
				tokenspos.append(pos)
		return (tokensword, tokenslemma, tokenspos)
	
	def remove_stopwords(self,sentence):
		wordlist = ['[Dd]an', '[Ww]eer', '[Ii]n', 'van', 'een', 'op','en','de','het','[Ii]k','jij','hij','zij','wij','jullie','deze','dit','die','dat','je', 'we','na','tot','te','hierin','onder']
		for x in wordlist:
			sentence = re.sub(' '+x+' ',' ', sentence)
			sentence = re.sub('\A'+x+' ',' ', sentence)
			sentence = re.sub(' '+x+'\Z',' ', sentence)
		return sentence

m = TopicExtraction()
#m.tryout(self.short_tweets)
m.tryout(m.tweets)
#m.read_from_file()
m.testout()
#m.get_postags()
