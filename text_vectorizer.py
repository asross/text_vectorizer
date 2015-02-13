"""
This is a module to convert labeled blobs of text to feature vectors suitable for more advanced NLP.

It stems all the words in all the text blobs, generates a list of 1- and 2-grams from them, excludes all the uncommon ones (< 3 occurrences),
then orders the 1- and 2-grams by frequency and converts each blob of text to a vector of which bigrams they contain (using the index into the sorted list)
with the count.

This can then be converted into a scipy sparse CSR matrix.

>>> list(bigrams(['foo', 'bar', 'baz']))
[('foo',), ('bar',), ('baz',), ('foo', 'bar'), ('bar', 'baz')]

>>> suffix('foo.csv', 'bar')
'foo_bar.csv'

>>> import os; os.system("echo '0,found founded founder bargain bargains bargainer bargained found bargain found bargain barge found bargain' > /tmp/__text_vectorizer_test.csv")
0

>>> vectorize_text('/tmp/__text_vectorizer_test.csv')
stemming text...
counting bigrams...
vectorizing text...
'/tmp/__text_vectorizer_test_stemmed_vectorized.csv'

>>> open('/tmp/__text_vectorizer_test_stemmed.csv').read().rstrip()
'0,found found found bargain bargain bargain bargain found bargain found bargain barg found bargain'

>>> map(lambda s: s.rstrip(), open('/tmp/__text_vectorizer_test_stemmed_bigram_counts.csv').readlines())
['bargain,7', 'found,6', 'found|bargain,4', 'bargain|bargain,3']

>>> open('/tmp/__text_vectorizer_test_stemmed_vectorized.csv').read().rstrip()
'0,2|0|3|1,4|7|3|6'

"""

import re
import csv
from collections import defaultdict
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

def bigrams(words):
  for word in words:
    yield (word,)
  i = 1
  while i < len(words):
    yield (words[i-1], words[i])
    i += 1

def suffix(csv_filename, ending):
  return csv_filename.replace('.csv', '_'+ending+'.csv')

def vectorize_text(csv_filename):
  print 'stemming text...'
  stemmed_filename = stem_text(csv_filename)
  print 'counting bigrams...'
  bigram_filename = count_bigrams(stemmed_filename)
  print 'vectorizing text...'
  return vectorize_bigrams(stemmed_filename, bigram_filename)

def stem_text(csv_filename):
  stemmer = LancasterStemmer()
  out_filename = suffix(csv_filename, 'stemmed')

  with open(csv_filename, 'rb') as input_file:
    with open(out_filename, 'wb') as output_file:
      output_csv = csv.writer(output_file)
      for row in csv.reader(input_file):
        label, text = row
        stemmed_uncommon_words = [stemmer.stem(word.lower()) for word in text.split() if not word.lower() in stopwords.words('english')]
        output_csv.writerow([label, ' '.join(stemmed_uncommon_words)])

  return out_filename

def count_bigrams(stemmed_filename):
  counts = defaultdict(int)

  with open(stemmed_filename, 'rb') as input_file:
    for row in csv.reader(input_file):
      label, stemmed_text = row
      for bigram in bigrams(stemmed_text.split()):
        counts[bigram] += 1

  out_filename = suffix(stemmed_filename, 'bigram_counts')

  with open(out_filename, 'wb') as output_file:
    output_csv = csv.writer(output_file)
    for bigram, count in sorted(counts.items(), key=lambda x: -x[-1]):  # sort them in descending order of frequency
      if count >= 3:
        output_csv.writerow(['|'.join(bigram), count])

  return out_filename

def vectorize_bigrams(stemmed_filename, bigram_filename):
  bigram_indices = {}
  out_filename = suffix(stemmed_filename, 'vectorized')

  with open(bigram_filename, 'rb') as count_file:
    i = 0
    for row in csv.reader(count_file):
      bigram, count = row
      bigram = tuple(bigram.split('|'))
      bigram_indices[bigram] = i
      i += 1

  with open(out_filename, 'wb') as output_file:
    output_csv = csv.writer(output_file)
    with open(stemmed_filename, 'rb') as input_file:
      for row in csv.reader(input_file):
        label, stemmed_text = row
        feature_vector_data = []
        feature_vector_indices = []
        bigram_counts = defaultdict(int)
        for bigram in bigrams(stemmed_text.split()):
          if bigram in bigram_indices:
            bigram_counts[bigram] += 1
        for bigram, count in bigram_counts.items():
          feature_vector_data.append(str(bigram_indices[bigram]))
          feature_vector_indices.append(str(count))
        output_csv.writerow([label, '|'.join(feature_vector_data), '|'.join(feature_vector_indices)])

  return out_filename

if __name__ == "__main__":
    import doctest
    doctest.testmod()
