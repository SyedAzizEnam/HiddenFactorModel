import pandas as pd

city = 'Phoenix'

df = pd.read_csv('../Data/yelp_academic_dataset_business.csv')

df = df[df['city']==city]

#df =df[df['categories'].map(lambda x: 'Restaurant' in x)]

with open('../Data/Phoenix_business_ids.txt', 'w') as f:
	for business_id in df['business_id']:
		f.write(str(business_id)+"\n")