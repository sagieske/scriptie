import csv
import gensim
import nltk
import re

class TopicExtraction_LDA(object):
	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated_v2.csv", 'rU'), delimiter=DELIMITER)
	class_dict = {"Y": 0, "N": 1, "U": 0}

	short_tweets = []
	tweets = []
	posdeleted_tweettokens = []
	posdeleted_tweets = []
	posdeleted_pos = []
	tuples = []
	pos_short_tweets = []
	pos_short_pos = []
	posdeleted_postypes = []
	dictionary_pos = {}

	PORTNUMBER = 1161

	EMOTICONS = ['[=:]-*[()DdPpSsOo(\|)(\$)]']
	NUMBERS = ['\d+[(\.):]\d+', '\d+']

	TOPICFILE = "day_output_class.csv"
	all_tweets = {}

	def __init__(self, column_tweet, column_class):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		DELIMITER = "\t"
		self.load_stopword_file()
		# Get activity tweets

		data = csv.reader(open(self.TOPICFILE, 'rU'), delimiter=DELIMITER)
		for i, row in enumerate(data):
			if (row[column_class] == '0'):
				self.all_tweets[i] = row[column_tweet]

	def gensim_test(self):
		tokenarray = []		
		for index in self.all_tweets:
			tweet_ascii = filter(lambda x: ord(x) < 128, self.all_tweets[index])
			cleaned_tweet = self.remove_stopwords(tweet_ascii.lower())
			tokenitem = nltk.word_tokenize(cleaned_tweet)
			tokenarray.append(tokenitem)

		#"""
		dictionary = gensim.corpora.Dictionary(tokenarray)

		dictionary.save('/tmp/testing_gensim.dict')


		corpus_mm = [dictionary.doc2bow(tokens_tweet) for tokens_tweet in tokenarray]
		gensim.corpora.MmCorpus.serialize('/tmp/testing_gensim.mm', corpus_mm) # store to disk, for later use

		tryout_topics = 10
		lda = gensim.models.ldamodel.LdaModel(corpus=corpus_mm, id2word=dictionary, num_topics=tryout_topics, update_every=1, chunksize=10000, passes=3)
 		hdp = gensim.models.hdpmodel.HdpModel(corpus_mm, id2word=dictionary)

		lsi = gensim.models.lsimodel.LsiModel(corpus=corpus_mm, id2word=dictionary, num_topics=tryout_topics)


		for i in range(0, tryout_topics):
			string = "topic #%i " + str(lda.print_topic(i))
			print string % i

		#for i in range(0, tryout_topics):
		#	string = "topic #%i " + str(lsi.print_topic(i))
			#print string % i

		#hdp.optimal_ordering()
		#hdp.print_topics()

		#"""

	def remove_stopwords(self,sentence):
		""" Removes stop words in sentences. Returns substituted sentence """
		"""
		wordlist = ['[Vv]anavond','[Ii]k', '[Jj]ij', '[Hh]ij', '[Ww]ij', '[Zz]ij', '[Jj]ullie','[Mm]ij', '[Mm]e', '[Jj]e', '[Zz]e', '[Ww]e', '[Ww]eer', '[Ll]ekker', '[Mm]aar', '[Ee]ens', '[Dd]e', '[Hh]et', '[Ee]en', '[Dd]an', '[Dd]aarom', '[Ww]aarom', '[Dd]us', '[Dd]at', '[Ff]f', '[Ww]at', '[eE]even', '[Dd]enk', '[Ee]ven', '[=:]-*[()DdPpSsOo(\|)(\$)]', '\d+[(\.):]\d+', '\d+', '[Ll]ekker']
		"""
		wordlist = []

		pattern_time = re.compile('vanavond|morgen|vandaag|vanmiddag|gister|gisteren|eerst|daarna')
		sentence = pattern_time.sub('', sentence)
		
		# append words like z'n 
		sentence = re.sub("'", "", sentence)
		sentence = re.sub("\W", " ", sentence)

		wordlist += self.STOPWORD_FILE + self.NUMBERS + self.EMOTICONS
		for x in wordlist:
			sentence = re.sub(' '+x+' ',' ', sentence)
			sentence = re.sub('\A'+x+' ',' ', sentence)
			sentence = re.sub(' '+x+'\Z',' ', sentence)

		# Delete 'laughs'
		sentence = re.sub('ha(ha)+',' ', sentence)

		# Delete 'links'
		#sentence = re.sub('(http://)(.*?)[(.com)(.nl)]',' ', sentence)
		#r = re.compile(r"([(http:)])(\w+)\b")
		#sentence = r.sub(' ', sentence)



		return sentence

	def load_stopword_file(self):
		stopword_file = open( "dutch-stop-words.txt", "r" )
		array = []
		for line in stopword_file:
			word = re.sub('\n','', line)
			array.append( word )
		self.STOPWORD_FILE = array
		



topicEX = TopicExtraction_LDA(3,5)
topicEX.gensim_test()
