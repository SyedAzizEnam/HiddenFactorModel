import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import string
from unidecode import unidecode

cachedstopwords = stopwords.words('english')

wnl = WordNetLemmatizer()

stemmer = SnowballStemmer('english')

def cleantext(text):
	''' Removes punctuation and converts to lowercase'''
	text = unidecode(text)
	unpuncted_text = reduce(lambda s,c: s.replace(c, ''), string.punctuation, text.lower()) 
	cleaned_text = ' '.join(word for word in unpuncted_text.split() if word not in cachedstopwords)
	stemmed_text = ' '.join(stemmer.stem(unidecode(wnl.lemmatize(wnl.lemmatize(word),'v'))) for word in cleaned_text.split())
	return stemmed_text


with open('restaurant_business_ids.txt','r') as f:
	business_ids = f.readlines()
	business_ids = map(lambda x: x.rstrip(), business_ids)

for j in range(1,6):
	review_bog = [] 

	reviews =pd.read_csv('yelp_academic_dataset_review{0}.csv'.format(j),encoding="utf-8")

	restaurant_reviews = reviews[reviews['business_id'].isin(business_ids)]
	ratings = restaurant_reviews['stars']
	review_text = restaurant_reviews['text']
	review_ ids = reviews['business_id']

	i=0
	for entry in review_text:
		words = {}
		cleaned_text = cleantext(entry)
		for word in cleaned_text.split():
			if word in words:
				words[word] += 1
			else:
				words[word] = 1
		review_bog.append(words)
		i+=1
		print "review: {0}".format(i)

	vocab = {}
	for data in review_bog:
		for key,value in data.iteritems():
			if key in vocab:
				vocab[key] += value
			else:
				vocab[key] = value

	with open('temp/vocab{0}.txt'.format(j),'w') as f:
		for key in sorted(vocab.iterkeys()):
			f.write("{0},{1}\n".format(key, vocab[key]))

	with open('temp/review_bog{0}.txt'.format(j),'w') as f:
		for review in review_bog:
			for k,v in review.iteritems():
				f.write("{0},{1} ".format(k,v))
			f.write("\n")
