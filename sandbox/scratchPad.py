import json
import pandas as pd

review_fileName = '../Data/yelp_academic_dataset_review.json'
business_fileName = '../Data/yelp_academic_dataset_business.json'
user_fileName = '../Data/yelp_academic_dataset_user.json'

with open(review_fileName) as f:
    reviews = pd.DataFrame(json.loads(line) for line in f)
f.close()
reviews = reviews.set_index(['review_id'])
reviews = reviews[['business_id', 'stars', 'text', 'user_id']]
reviews.to_csv('../Data/reviews.csv', sep='\t', encoding='utf-8', line_terminator='\nsen\n')

with open(business_fileName) as f:
	business = pd.DataFrame(json.loads(line) for line in f)
f.close()
business = business.set_index(['business_id'])

business_categories = []
for c in business['categories']:
	business_categories += c
business_categories = list(set(business_categories))
business_categories.sort()
f= open('../Data/business_categories.txt', 'w')
f.writelines(["%s\n" % str(category)  for category in business_categories])
f.close()

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

sentence = reviews[:1]['text'].values
words = nltk.word_tokenize(sentence)

filtered_words = [w for w in words if w not in stopwords.words('english')]
stemmer = SnowballStemmer('english')
stemmed_words = [stemmer.stem(w) for w in filtered_words]

def process_text(sentence, stemmer=SnowballStemmer('english'), stopwords=set(stopwords.words('english'))):
	return [stemmer.stem(w) for w in nltk.word_tokenize(sentence.lower()) if w.isalpha() and w not in stopwords]

def process_text(sentence, stemmer=SnowballStemmer('english'), lmtzr = nltk.stem.wordnet.WordNetLemmatizer(), stopwords=set(stopwords.words('english'))):
		sentence = nltk.word_tokenize(sentence.lower())		#Tokenize
		words = [w for w in sentence if w.isalpha() and w not in stopwords]	#Remove stopwords
		return [stemmer.stem(lmtzr.lemmatize(w)) for w in words]		#Stem and Lemmatize

reviews['text'] = reviews['text'].apply(process_text)


for i,entry in enumerate(reviews['text']):
    try:
        len(entry)
    except:
        print i, entry

f = open('../Data/stemmed_reviews.csv', 'rb')
reader = csv.reader(f)
business_ids = []
for row in reader:
	business_ids += [row[1]]
business_ids = business_ids[1:]
business_ids = list(set(business_ids))
restaurant_ids = set([id.strip() for id in open('../Data/restaurant_business_ids.txt', 'rb')])
for id in business_ids:
	if id not in restaurant_ids:
		print id