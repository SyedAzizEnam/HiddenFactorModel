import json
import pandas as pd
import nltk
from sqlalchemy import create_engine
from datetime import datetime as dt
import csv
from os import path

class reviewProcessor :

	def __init__(self, review_fileName):

		self.stopwords = set(nltk.corpus.stopwords.words('english'))
		self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
		self.lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
		self.review_fileName = review_fileName
		self.disk_engine = create_engine('sqlite:///..Data/yelp.db')
		self.restaurant_ids = set([id.strip() for id in open('../Data/restaurant_business_ids.txt', 'rb')])


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
		chunkSize = 20000
		
		records = []
		with open(self.review_fileName, 'rb') as csvfile:
			reader = self.unicode_csv_reader(csvfile, delimiter='\t')
			for row in reader:
				if i == 0:
					row = [field.encode('ascii', 'ignore') for field in row]
					schema = row
					text_ix = schema.index('text')
					businessID_ix = schema.index('business_id')
				else:
					if i%2 == 0:
						if str(row[businessID_ix]) in self.restaurant_ids:
							review_text = row[text_ix]
							row[text_ix] = self.process_text(review_text)
							row = [field.encode('ascii', 'ignore') for field in row]
							if len(row[text_ix]) > 0:
								records += [row]
							if len(records)%chunkSize == 0:
								print '{} seconds: completed {} rows'.format((dt.now() - time_start).seconds, len(records))
				i += 1

		csvfile.close()

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
		chunkSize = 20000
		
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
					businessID_ix = schema.index('business_id')
					records += [schema]

				else:
					if i%2 == 0:
						if row[businessID_ix] in self.restaurant_ids:
							review_text = row[text_ix]
							row[text_ix] = self.process_text(review_text)
							row = [field.encode('ascii', 'ignore') for field in row]
							if len(row[text_ix]) > 0:
								records += [row]
							if len(records)%chunkSize == 0:
								chunkNum += 1
								writer.writerows(records)
								print '{} seconds: completed {} rows out of {} reviews'.format((dt.now() - time_start).seconds, chunkNum*chunkSize, i/2)
								records = []
				i += 1

			if len(records) > 0:
				writer.writerows(records)
				print '{} seconds: completed {} rows out of {} reviews'.format((dt.now() - time_start).seconds, chunkNum*chunkSize+len(records), i/2)
				
			outputFile.close()

		csvfile.close()


if __name__ == "__main__":
	review_fileName = '../Data/yelp_academic_dataset_review.json'

	if not path.exists('../Data/reviews.csv'):
		print 'Loading json file into pandas dataframe...'
		with open(review_fileName) as f:
			reviews = pd.DataFrame(json.loads(line) for line in f)
		f.close()
		print 'Finished loading json file into pandas dataframe'

		print 'Projecting relevant columns: review_id (index), business_id, stars, text, user_id...'
		reviews = reviews.set_index(['review_id'])
		reviews = reviews[['business_id', 'stars', 'text', 'user_id']]

		print 'Projection completed... starting write into csv file...'
		reviews.to_csv('../Data/reviews.csv', sep='\t', encoding='utf-8', line_terminator='\nsen\n')
		print 'Finished writing into csv file: ../Data/reviews.csv'

	print 'Beginning processing of reviews to perform stemming and lemmatization of all words...'
	review_fileName = '../Data/reviews.csv'
	reviewer = reviewProcessor(review_fileName)
	reviewer.process_save('../Data/processed_reviews.csv')
	print 'Completed processing'