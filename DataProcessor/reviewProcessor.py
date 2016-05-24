import json
import pandas as pd
import nltk
from sqlalchemy import create_engine
from datetime import datetime as dt
import csv

class reviewProcessor :

	def __init__(self):

		self.stopwords = set(nltk.corpus.stopwords.words('english'))
		self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
		self.lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
		self.review_fileName = '../Data/reviews.csv'
		self.disk_engine = create_engine('sqlite:///..Data/yelp.db')


	def process_text(self, sentence):
		sentence = nltk.word_tokenize(sentence.lower())		#Tokenize
		words = [w for w in sentence if w.isalpha() and w not in self.stopwords]	#Remove stopwords
		stemmed_words = [self.stemmer.stem(self.lmtzr.lemmatize(w)) for w in words]		#Stem and Lemmatize
		return ' '.join(stemmed_words)


	def unicode_csv_reader(self, utf8_data, dialect=csv.excel, **kwargs):
	    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
	    for row in csv_reader:
	        yield [unicode(cell, 'utf-8') for cell in row]


	def process(self):

		time_start = dt.now()
		i = 0
		
		records = []
		with open(self.review_fileName, 'rb') as csvfile:
			reader = self.unicode_csv_reader(csvfile, delimiter='\t')
			for row in reader:
				if i == 0:
					row = [field.encode('ascii', 'ignore') for field in row]
					schema = row
					text_ix = schema.index('text')
				else:
					if i%2 == 0:
						review_text = row[text_ix]
						row[text_ix] = self.process_text(review_text)
						row = [field.encode('ascii', 'ignore') for field in row]
						records += [row]
						if len(records)%2000 == 0:
							print '{} seconds: completed {} rows'.format((dt.now() - time_start).seconds, len(records))
				i += 1

		self.schema = schema
		self.records = records


	def save(self, output_fileName):

		outputFile = open(output_fileName, 'wb')
		writer = csv.writer(outputFile)
		writer.writerow(self.schema)
		writer.writerows(self.records)
		outputFile.close()


	def process_save(self, output_fileName):

		time_start = dt.now()
		i = 0
		chunkSize = 2000
		
		records = []
		with open(self.review_fileName, 'rb') as csvfile:
			reader = self.unicode_csv_reader(csvfile, delimiter='\t')

			outputFile = open(output_fileName, 'ab')
			writer = csv.writer(outputFile)

			chunkNum = 0
			for row in reader:

				if i == 0:	#Happens only once
					row = [field.encode('ascii', 'ignore') for field in row]
					schema = row
					text_ix = schema.index('text')
					records += [schema]

				else:
					if i%2 == 0:
						review_text = row[text_ix]
						row[text_ix] = self.process_text(review_text)
						row = [field.encode('ascii', 'ignore') for field in row]
						records += [row]
						if len(records)%2000 == 0:
							chunkNum += 1
							writer.writerows(records)
							print '{} seconds: completed {} rows'.format((dt.now() - time_start).seconds, chunkNum*chunkSize)
							records = []
				i += 1

			if len(records) > 0:
				writer.writerows(records)
				
			outputFile.close()