import pandas as pd


def buildVocab(text):
    """Takes a dataframe of texts and creates a vocab file
    vocab.txt"""

    vocab = {}
    for entry in text:
        # if isinstance(entry, str) == False:
        #     continue
        for word in entry.split():
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1

    filtered_vocab = {}
    for key in vocab.iterkeys():
        if vocab[key] >= 100:
            filtered_vocab[key] = vocab[key]

    with open('../Data/vocab.txt', 'w') as f:
        i = 0
        for key in sorted(filtered_vocab.iterkeys()):
            f.write("{0},{1},{2}\n".format(i, key, filtered_vocab[key]))
            i += 1


def bagOfWord(text, vocab_lookup):
    counts = {}

    try:
        for word in text.split():
            if word in counts:
                counts[word] += 1
            elif word in vocab_lookup:
                counts[word] = 1
    except:
        print text

    return ' '.join(str(vocab_lookup[word]) + ":" + str(counts[word]) for word in counts.iterkeys())


if __name__ == "__main__":

    print 'Loading csv file into pandas dataframe...'
    reviews = pd.read_csv('../Data/processed_reviews.csv')
    reviews = reviews.set_index(['review_id'])
    review_text = reviews['text']  # .apply(lambda row: str(row))
    print 'Finished loading'

    print 'Building vocabulary...'
    buildVocab(review_text)
    print 'Complete'

    print 'Loading dictionary...'
    vocab_lookup = {}
    with open('../Data/vocab.txt', 'r') as f:
        for line in f.readlines():
            lookup_int, key, __ = line.split(',')
            vocab_lookup[key] = lookup_int
    print 'Complete'

    print 'Building bag of words...'
    reviews['text'] = review_text.apply(lambda row: bagOfWord(row, vocab_lookup))
    reviews = reviews[reviews['text'] != '']
    print 'Finished...'

    print 'Saving bag of words into file...'
    reviews.to_csv('../Data/bow_reviews.csv')
    print 'Process complete!'
