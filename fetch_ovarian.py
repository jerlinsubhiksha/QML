import urllib.request
import arff
import pandas as pd

print('Downloading Real Ovarian Dataset from OpenML...')
url = 'https://openml.org/data/v1/download/22112159/Ovarian.arff'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla'})
response = urllib.request.urlopen(req)
content = response.read().decode('utf-8')

print('Parsing ARFF format with liac-arff...')
dataset = arff.loads(content)
df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

out_path = r'C:\Users\hp\Documents\antigravity\optimistic-meitner\qml-biobank-research\data\raw\real_ovarian_cancer.csv'
df.to_csv(out_path, index=False)
print(f'Successfully saved to {out_path}!')
print('Shape:', df.shape)
