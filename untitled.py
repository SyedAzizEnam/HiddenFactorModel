with open('yelp_academic_dataset_review.json') as f:
	header = f.readline()
	with open('yelp_academic_dataset_review1.json','w')as g:
		g.write(header)
		for x in xrange(0,500000):
			g.write(f.readline())
	with open('yelp_academic_dataset_review2.json','w')as g:
		g.write(header)
		for x in xrange(0,500000):
			g.write(f.readline())
	with open('yelp_academic_dataset_review3.json','w')as g:
		g.write(header)
		for x in xrange(0,500000):
			g.write(f.readline())
	with open('yelp_academic_dataset_review4.json','w')as g:
		g.write(header)
		for x in xrange(0,500000):
			g.write(f.readline())
	with open('yelp_academic_dataset_review5.json','w')as g:
		g.write(header)
		for x in xrange(0,200000):
			g.write(f.readline())