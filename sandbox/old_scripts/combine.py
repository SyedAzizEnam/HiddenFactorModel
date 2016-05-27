vocab = {}

for i in range(1,6):
	with open('temp/vocab{0}.txt'.format(i)) as f:
		for line in f:
			word, count=line.rstrip().split(',')
			if int(count) >=100:
				vocab[word] = int(count)

vocab_lookup = {}

with open('vocab.txt','w') as f:
	i=0
	for key in sorted(vocab.iterkeys()):
		vocab_lookup[key]= i 
		f.write("{0},{1},{2}\n".format(i, key, vocab[key]))
		i +=1

for i in range(1,6):
	with open('temp/review_bog{0}.txt'.format(i)) as f:
		with open('review.txt','w') as g:
			for line in f: 
				word_count=line.rstrip().split()
				for entry in word_count:
					word, count = entry.split(',')
					if word in vocab:
						g.write("{0},{1} ".format(vocab_lookup[word], int(count)))
				g.write("\n")
