# File for help functions
import pickle
import csv

def dump_to_file(filename, dumpobject):
	"""	Dump array to file """
	f = file(filename, "w")
	pickle.dump(dumpobject, f)

def unfold_tuples(dictionary):
	""" Unfold all tuples in dictionary to single items"""
	unfold_tuples_array = []
	for x in dictionary:
		for item in x:
			unfold_tuples_array.append( item )
	list(set(unfold_tuples_array))
	return unfold_tuples_array

def unfold_tuples_strings(array):
	""" Unfold all tuples in dictionary to strings"""
	unfold_tuples_strings = []
	for tupleset in array:
		string = ''
		for index, word in enumerate(tupleset):
			if index != len(tupleset)-1:
				string += word + ' '
			else:
				string += word
		unfold_tuples_strings.append(string)
	return unfold_tuples_strings

def write_to_csv(filename, write_type, row_array):
	""" Write arrays to CSV file """
	openfile  = open(filename, write_type)
	csv_writer = csv.writer(openfile,delimiter = ',')
	for row in row_array:
		csv_writer.writerow(row)
	openfile.close()

def read_from_file(filename):
	"""	Load array from file """
	f = file(filename, "r")
	pickle_object = pickle.load(f)
	return pickle_object

def write_classification_to_tweetfile(classificationarray,startrow, startcolumn, inputfile, outputfile, special):
	""" Write classification for tweet in csv file containing tweets """
	DELIMITER = "\t"
	newdata = csv.reader(open(inputfile, 'rU'), delimiter=DELIMITER)
	output = csv.writer(open(outputfile, 'wb'), delimiter=DELIMITER)

	indexarray = 0

	for i,row in enumerate(newdata):
		if (i < startrow ) :
			output.writerow(row)
		else:
			if (special and int(row[startcolumn-1]) == 1):
				output.writerow(row)
			else:
				newrow = row[:startcolumn]
				if ( isinstance( classificationarray[indexarray], list) ):
					newrow += classificationarray[indexarray]
				else:
					if ( isinstance( classificationarray[indexarray], str) ):
						newrow.append(classificationarray[indexarray])
					else: 
						newrow = row
				output.writerow(newrow)
				indexarray += 1
 	
