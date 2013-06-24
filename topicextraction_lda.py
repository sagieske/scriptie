import csv
import gensim
import nltk
import re
from topicextraction_dictionary import TopicExtraction_dictionary
import operator

class TopicExtraction_LDA(object):
	# Read file
	DELIMITER = "\t"
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


	EMOTICONS = ['[=:]-*[()DdPpSsOo(\|)(\$)]']
	NUMBERS = ['\d+[(\.):]\d+', '\d+']

	TOPICFILE = "day_saturday_class.csv"
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
		print "INIT DONE"


	def gensim_test(self):

		topiclogObject = TopicExtraction_dictionary("","day_saturday_class.csv", "day_friday_class.csv")
		
		f = open('loglikelihood.txt','w')
		topWordsDict = topiclogObject.get_top_likelihood(topiclogObject.loglikelihood, 500)
		
		topwordsTOFILE = sorted(topWordsDict.iteritems(), key=operator.itemgetter(1), reverse=True)
		heading = "TOTAL words in corpus: " +str(sum(topiclogObject.corpus.values())) +".\nTOTAL words in reference: " +str(sum(topiclogObject.referencecorpus.values())) +"\nTOTAL messages of corpus: " + str(len(topiclogObject.corpus))  +"\nTOTAL messages of reference corpus: " + str(len(topiclogObject.referencecorpus))  + "-"*40  + "\nWORD".ljust(20) + " LLR".ljust(10) +  "CORPUS".ljust(10) + "REFERENCE".ljust(10) + "\n"

		f.write(heading)
		for item in topwordsTOFILE:
			llr_str = "%.4f" %item[1]
			string = item[0].encode("utf-8").ljust(20) + llr_str.ljust(10) + str(topiclogObject.corpus[item[0]]).ljust(10) + str(topiclogObject.referencecorpus.get(item[0],0)).ljust(10) + "\n"
			f.write(string)
		f.close
		print "JAHO"


		topWords = topWordsDict.keys()
		tokenarray = []		
		for index in self.all_tweets:
			tweet_ascii = filter(lambda x: ord(x) < 128, self.all_tweets[index])
			#cleaned_tweet = self.remove_stopwords(tweet_ascii.lower())
			#tokenitem = nltk.word_tokenize(cleaned_tweet)
			tokenitem = nltk.word_tokenize(tweet_ascii)
			new_tokenitem = []
			for token in tokenitem:
				if token in topWords:
					new_tokenitem.append(token)

			#if ' naar ' in cleaned_tweet:
			#	index_nr = tokenitem.index('naar')
			#	print self.all_tweets[index]
			#	print ' '.join(tokenitem[index_nr:index_nr+2])

			tokenarray.append(new_tokenitem)

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
		r = re.compile(r"[(http://)www.][^ ]*")
		sentence = r.sub(' ', sentence)

		return sentence

	def load_stopword_file(self):
		stopword_file = open( "dutch-stop-words.txt", "r" )
		array = []
		for line in stopword_file:
			word = re.sub('\n','', line)
			array.append( word )
		self.STOPWORD_FILE = array

	def tryout(self):
		naartweets = []
		for key in self.all_tweets:
			if 'naar' in self.all_tweets[key]:
				naartweets.append(self.all_tweets[key])

		for item in naartweets:
			tokens = nltk.word_tokenize(item)
			for index, word in enumerate(tokens):
				if word == 'naar':
					print tokens[index:index+4]
					print item
		print len(naartweets)

		



topicEX = TopicExtraction_LDA(3,5)
topicEX.gensim_test()
#topicEX.tryout()
