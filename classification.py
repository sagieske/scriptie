import csv
import sys

class Main():

	# Lists for tweet and class
	tweets = {}
	tweet_class = {}
	
	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 1}

	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test.csv", 'rU'), delimiter=DELIMITER)

	def __init__(self):
		# Load tweets from file
		self.createSets()
		self.count_classes()

	def createSets(self):
		"""
		Creates training and test set
		"""
		print "Initialize sets.."
		# Create tweet and class lists
		for i, row in enumerate(self.data):
			# Ignores header
			#if(i != 0):
			# TEMP, for testing only
			if (i < 20):
				# Get tweet and class 
				self.tweets[i-1] = row[3]
				self.tweet_class[i-1] = self.class_dict.get(row[5])

	
	def count_classes(self):
		"""
		Counts and prints occurance of each class
		"""
		values = self.tweet_class.values()
		total = len(values)
	
		# Count occurances of classes
	 	activity_count = values.count(0)
		nonactivity_count = values.count(1)

		# Print
		print ">> Statistics:"
		print "Total number of tweets: %i" % total
		print "Total activity tweets: %i" % activity_count
		print "Total non-activity tweets: %i" % nonactivity_count
	

m = Main()
