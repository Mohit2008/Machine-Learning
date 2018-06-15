import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import seaborn as sns
import pickle

sns.set()
np.random.seed(2018)

docword = pd.read_csv("docword.nips.txt", nrows=3, header=None) # read the data with top 3 rows

with open("vocab.nips.txt") as g:
    vocab = g.read().split("\n")  # vocabulary file

no_of_docs = int(docword.iloc[0,])  # get the no of documents
no_of_vocab_words = int(docword.iloc[1,])  # get no of vocablary words
no_of_words = int(docword.iloc[2,])  # get the no of words
no_of_topics = 30  # no of topics to be clustered into

smooth_value = 0.00001 # set smoothing criteria
threshold = 0.0001  # set threshold

docword = np.array(pd.read_csv("docword.nips.txt", sep=" ", skiprows=[0, 1, 2, -1], header=None)) # read the data without top 3 rows
document_vector = np.zeros((1500, 12419)) # initialize the vector of documnets
for i in range(np.shape(docword)[0]):
    document_vector[int(docword[i, 0] - 1), int(docword[i, 1] - 1)] = float(docword[i, 2]) # set the word count for each document

weight = np.zeros((no_of_docs, no_of_topics)) # initialize a weight array
pij = np.zeros(no_of_topics) # initialize initial prob
random_labels=np.random.randint(0,30, size=1500) # randomly generate labels

for i in range(no_of_topics):
    pij[i] = Counter(random_labels)[i] / document_vector.shape[0] # set the pij array

temp = np.zeros((no_of_topics, np.shape(document_vector)[1]))
# count all the words for each label
for i, label in enumerate(random_labels):
    for j in range(no_of_topics):
        if label == j:
            temp[j,] += document_vector[i,] # store the word count for each label

# normalize the counts that have 0
for i in range(no_of_topics):
    for j in range(np.shape(document_vector)[1]):
        if temp[i, j] == 0:
            temp[i, j] += 0.05 # smoothing

pjk = np.zeros((no_of_topics, np.shape(document_vector)[1]))
pjk = temp / np.sum(temp, axis=1)[:, None]
temp1 = np.zeros((no_of_docs, no_of_topics))
############### Estimation##############################################################################################
qs = []  # Empty list to hold Q
qs.append(0)  # First value of Q
print("E-M step started")
for k in range(1000):
    for i in range(no_of_docs):
        for j in range(no_of_topics):
            temp1[i, j] = np.sum(document_vector[i,] * np.log(pjk[j,])) + np.log(pij[j])
        weight = (np.exp(temp1 - temp1.max(1)[::, None])) / (np.exp(temp1 - temp1.max(1)[::, None])).sum(1)[::, None]

    ################Maximisation############################################################################################
    pjk_norm = np.zeros((no_of_topics, document_vector.shape[1]))
    for i in range(no_of_topics):
        for j in range(no_of_docs):
            pjk_norm[i,] += np.dot(document_vector[j,], weight[j, i])
    pjk_norm += smooth_value  # prevent 0 word probability
    pjk_2 = np.sum(pjk_norm, axis=1) + smooth_value * document_vector.shape[1]
    pjk = pjk_norm / pjk_2[::, np.newaxis]

    pij = np.sum(weight, axis=1) / no_of_docs

    ##############Convergnce################################################################################################
    q1 = 0
    for i in range(no_of_docs):
        for j in range(no_of_topics):
            q1 += (np.dot(document_vector[i,], np.log(pjk[j, :])) + np.log(pij[j])) * weight[i, j]
    qs.append(q1)
    print(q1)
    # check threshold
    if np.abs(qs[k] - qs[k - 1]) < threshold:
        break

#######Generate the prob of each topic
print("Generating topic probability")
topic = np.zeros(no_of_docs)
prob = np.zeros(30)
for i in range(no_of_docs):
    topic[i] = np.argsort(weight[i,])[::-1][0]
    for j in range(no_of_topics):
        if topic[i] == j:
            prob[j] += 1
prob = prob / no_of_docs
prob_percent=[i*100 for i in prob]
plt.plot([i for i in range(1,31)], prob)
plt.xlabel('Topic no')
plt.ylabel('Probability')
plt.title("Probability Chart")
plt.tight_layout()
plt.savefig("Topic_prob.png")  # save the plot
plt.close()  # close the plot

#######Top 10 words for each topic
print("Generating top 10 words")
index = []
word_count = np.zeros((no_of_topics, 10))

for i in range(no_of_topics):
    index.append(np.argsort(pjk[i,])[::-1][0:10])

word_count = np.zeros((30, 10), dtype=object)
for i in range(len(index)):
    for j in range(10):
        word_count[i][j] = vocab[index[i][j]]

freq_word_df = pd.DataFrame(word_count)
freq_word_df.to_csv("topic_words.csv")

