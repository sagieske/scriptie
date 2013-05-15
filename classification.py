import csv
import sys

def createSets(filename):
	"""
	Creates training and test set
	Input: filename
	"""
	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	# Read file
	DELIMITER = ","
	data = csv.reader(open(filename, 'rU'), delimiter=DELIMITER)
	#totaltweets = len(data)
	print data[22]
	#for i, row in enumerate(data):

		

def main(argv):
	"""
	Program entry point.
	Input: datafile
	"""
	createSets(argv[1])
	


if __name__ == '__main__':
	sys.exit(main(sys.argv))
