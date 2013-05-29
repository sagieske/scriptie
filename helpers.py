# File for help functions

def unfold_tuples(dictionary):
	""" Unfold all tuples in dictionary to single items"""
	unfold_tuples_array = []
	for x in dictionary:
		for item in x:
			unfold_tuples_array.append( item )
	list(set(unfold_tuples_array))
	return unfold_tuples_array
