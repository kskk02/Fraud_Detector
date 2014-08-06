import pandas as pd

def GetData(filename):
	"""
	Description: This gets the data from a raw call log file.
	INPUT: 
	       filename  - type: string - This is the filename that has the log file.
	OUTPUT: Returns a dataframe containing the processed log file.
	"""

	# this function returns the dataset and removed any rows where there are NA's in the caller or callee column
	df = pd.read_csv(filename, converters={'calldno': str,'callgno': str} )
	df= df.dropna(subset = ["callgno","callgno"])
	df.callgno.apply(lambda x : x.encode('utf-8'))
	df.calldno.apply(lambda x : x.encode('utf-8'))

	print df.shape
	return df