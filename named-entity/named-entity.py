import PyPDF2
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) # Getting the stop word list for english corpus
stem = nltk.PorterStemmer() # Initialising the stemmer

# Read the pdf file
def read_pdf(filename):
    content = ""
    pdfFileObj = open(filename, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    for x in range(0, pdfReader.numPages):
        pageObj = pdfReader.getPage(x)
        content += (pageObj.extractText()).strip()
    return content

# Do prepossessing with the text blob by splitting it , tokenizing it and doing Part of speech tagging
def process_data(pdf_content):
    sent_tok= pdf_content.split("\n") # split different sentences
    word_tok= [nltk.word_tokenize(w) for w in sent_tok] #split each sentence to its tokens
    word_tok=[w for w in word_tok if w !=[] and w not in [';','.',':']] # clean up some empty lists
    for inner_list in word_tok:
        for word in inner_list:
            if word in stop_words:
                inner_list.remove(word) # Remove stop words
    tags= [nltk.pos_tag(w) for w in word_tok] # tag each word
    return tags


pdf_content= read_pdf("sample.pdf")
tagged_list =process_data(pdf_content)
chunk_sentence = [nltk.ne_chunk(w) for w in tagged_list] # Do the named-entity using pre trained classifier
for sentence in chunk_sentence:
    entity=[]
    for word in sentence:
        ## This piece of code is copied from stack overflow to ease out the interpretation of the output
        if hasattr(word, 'label'):
            entity.append(word.label()+" "+' '.join([child[0] for child in word]))
    if len(entity)!=0:
        with open('output.txt', 'a') as file:
            file.write(str(entity)) # Outputting the n-e for each sentence
            file.write("\n")


##**Citations and Notes**

# Some of the nltk methods have been looked up in its documentation ,
            # was planning to use metapy library which is much faster but for that i would have had to train a classifier to do name-entity recognition"