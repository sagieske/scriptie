import csv
import sys

def createSets(filename):
	"""
	Creates training and test set
	Input: filename
	"""
	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	# Lists for tweet and class
	tweets = {}
	tweet_class = {}
	
	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 1}

	# Read file
	DELIMITER = "\t"
	data = csv.reader(open(filename, 'rU'), delimiter=DELIMITER)
	
	# Create tweet and class lists
	for i, row in enumerate(data):
		# Ignores header
		#if(i != 0):
		# TEMP, for testing only
		if (i < 20):
			# Get tweet and class 
			tweets[i-1] = row[3]
			tweet_class[i-1] = class_dict.get(row[5])

	count_classes(tweet_class)
	
def count_classes(tweet_class):
	"""
	Counts and prints occurance of each class
	"""
	values = tweet_class.values()
	total = len(values)
	
	# Count occurances of classes
 	activity_count = values.count(0)
	nonactivity_count = values.count(1)

	# Print
	print ">> Statistics:"
	print "Total number of tweets: %i" % total
	print "Total activity tweets: %i" % activity_count
	print "Total non-activity tweets: %i" % nonactivity_count
	

def main(argv):
	"""
	Program entry point.
	Input: datafile
	"""
	createSets(argv[1])
	


if __name__ == '__main__':
	sys.exit(main(sys.argv))
