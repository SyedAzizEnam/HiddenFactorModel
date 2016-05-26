import pandas as pd


def buildVocab(text):
	'''Takes a dataframe of texts and creates a vocab file
	vocab.txt'''

	vocab = {}
	for entry in text:
		if isinstance(entry, str)== False:
			continue
		for word in entry.split():
			if word in vocab:
				vocab[word] += 1
			else:
				vocab[word] = 1

	filtered_vocab = {}
	for key in vocab.iterkeys():
		if vocab[key] >= 100:
			filtered_vocab[key] = vocab[key]

	with open('../Data/vocab.txt','w') as f:
		i=0
		for key in sorted(filtered_vocab.iterkeys()):
			f.write("{0},{1},{2}\n".format(i, key, filtered_vocab[key]))
			i += 1

def bagOfWord(text,vocab_lookup):

	counts = {}

	for word in text.split():
		if word in counts:
			counts[word] += 1
		elif word in vocab_lookup:
			counts[word] = 1

	return ' '.join(str(vocab_lookup[word])+":"+str(counts[word]) for word in counts.iterkeys())



if __name__ == "__main__":

	reviews =pd.read_csv('../Data/processed_reviews.csv',encoding="utf-8")
	review_text = reviews['text'].apply(lambda row: str(row))

	buildVocab(review_text)

	vocab_lookup = {}
	with open('../Data/vocab.txt','r') as f:
		for line in f.readlines():
			lookup_int, key, __ = line.split(',')
			vocab_lookup[key] = lookup_int

	reviews['text'] = review_text.apply(lambda row: bagOfWord(str(row), vocab_lookup))

	reviews.to_csv('../Data/bow_reviews.csv',encoding='utf-8')

