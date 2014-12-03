


# read in the vocab list

f = open('vocab_formatted.txt')
vocab = [line.strip() for line in f]
f.close()

# read in the email line by line

email = []

f = open('testemail1')
[email.extend(line.strip().split()) for line in f]
f.close

#print email

# check each word of the email against the vocab list and build 
# up an array of word indices.  The index is the location of the word
# in the vocab list

wordIndices = []

for i in email:
  if i in vocab:
    wordIndices.append(vocab.index(i))

#print wordIndices

# feature vector length is equal to length of vocabulary list

vecLength = len(vocab)
featureVector = [0] * vecLength

for i in wordIndices:
  featureVector[i] = 1

#print len(featureVector)
#print sum(featureVector)
#print featureVector
for val in featureVector:
  print val
