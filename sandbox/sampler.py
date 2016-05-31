import numpy as np
from scipy.sparse import csr_matrix
import random

def loglikelihood(theta, phi, bow_reviews, topic_assingments):
	"""Computes likelihood of a corpus

	Args:
	theta: topic distribution per item. (IxK numpy array)
	phi: word distributions per topic. (KxV numpy array)
	boW_reviews: BOW reviews in sparse matrix.
	topic_assingments: topic assingments for each word in a review.
	(list of numpy arrays)

	Returns:
	loglikelihood: The likelihood of the corpus
	"""

	I = bow_reviews.getcol(0).toarray().size

	All_loglikelihoods = []

	for i in xrange(I):
		review = bow_reviews.getrow(i).toarray()[0]
		topics = topic_assingments[i]
		Nd = review.sum()
		words = flatten_BOW(review)

		loglikelihood = 0

		for word,topic in zip(words,topics):
			#print theta[i,topic],phi[topic,word]
			loglikelihood += np.log(theta[i,topic])+np.log(phi[topic,word])

		All_loglikelihoods.append(loglikelihood)

	return sum(All_loglikelihoods)

def sampleWithDistribution(p):
	""" Sampler that samples with respect to distribution p

	Args:
	p : numpy array

	Returns:
	i: index of sampled value
	"""
	r = random.random()  # Rational number between 0 and 1
	
	for i in xrange(len(p)):
		r = r - p[i]
		if r<=0:
			return i
	raise Exception("Uh Oh... selectWithDistribution with r value %f" %r)


def Gibbsampler(boW_reviews, phi, theta):
	"""
	Resamples the topic_assingments accross the entires corpus

	Args:
	boW_reviews: reviews in bow format
	theta : topic distribution per item. (IxK numpy array)
	phi: word distributions per topic. (KxV numpy array)

	Returns: 
	new_topic_assingments: list of numpy arrays
	"""


	I = bow_reviews.getcol(0).toarray().size

	new_topic_assingments = []
	
	for i in xrange(I):
		review = bow_reviews.getrow(i).toarray()[0]
		words = flatten_BOW(review)
		topic_assingments = []
		
		for word in words:
			p = theta[i,:]*phi[:,word]
			p = p/p.sum()
			topic_assingments.append(sampleWithDistribution(p))
		
		topic_assingments = np.array(topic_assingments)
		new_topic_assingments.append(topic_assingments)
	
	return new_topic_assingments

def flatten_BOW(bow_review):
	""" Flattens the bag of words array so that it is the
	same shape as the topic_assingment array

	Args:
	bow_review: numpy array 

	Returns:
	words: numpy array
	"""

	indicies = np.flatnonzero(bow_review)

	words = [] 

	for j in indicies:
		for k in xrange(bow_review[j]):
			words.append(j)

	return np.array(words)

if __name__ == "__main__":

	K=5

	data = np.array([1, 2, 3, 4, 5, 6])
	indptr = np.array([0, 2, 3, 6])
	indices = np.array([0, 2, 2, 0, 1, 2])

	bow_reviews = csr_matrix((data, indices, indptr), shape=(3, 3))
	
	theta = np.random.rand(3,K)
	theta = theta/theta.sum(axis=1)[:,None]

	phi = np.random.rand(K,3)
	for i in xrange(phi.shape[0]):
		phi[i,:] = phi[i,:]/phi[i,:].sum()

	topic_assingments = []

	for i in xrange(bow_reviews.shape[0]):
		topics = np.random.randint(K, size=bow_reviews.getrow(i).sum())
		topic_assingments.append(topics)
	
	kappa = 1.0

	print bow_reviews.toarray()
	print topic_assingments
	print theta
	print phi

	print loglikelihood(theta, phi, bow_reviews,topic_assingments)
	print Gibbsampler(bow_reviews, phi, theta)

