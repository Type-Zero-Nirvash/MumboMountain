import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

#####################################################################
# All example text used is from 'Banjo-Kazooie'
# A game for the Nintendo64 made by Rareware

lineOne = "Many tricks are up my sleeve, to save yourself, you\'d better leave!"

lineTwo = "I\'ve got this skirt so when I\'m thinner, it really makes me look a winner!"

lineThree = "Grunty admits she\'s a hog, I really need a big hot dog!"

lineFour = "You side with Banjo, but change tack, imagine you on Grunty\'s back!"

def makeTokens(text):

    print("\n")
    print(text)
    token = word_tokenize(text)
    print("\n")
    print(token)

    return token

def filterStops(token):

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    filtered_text = []
    for word in token:
        if word.casefold() not in stop_words:
            filtered_text.append(word)

    print("\n")
    print(filtered_text)

    return filtered_text

#####################################################################
# Tokenization

print("\n")
print(lineOne)

# Tokenized input with stop words
tokenized_data = word_tokenize(lineOne)
print(tokenized_data)

# Filter Stop Words
## Removes punctuation and common words that do not add meaning
## to the phrase as a whole
# Uncomment if stopwords has not been downloaded
# nltk.download("stopwords")
from nltk.corpus import stopwords

# Set stopwords
stop_words = set(stopwords.words("english"))

# Parse text and remove Stop Words
filtered_text = []
for word in tokenized_data:
    if word.casefold() not in stop_words:
        filtered_text.append(word)

# Without Stop Words
print("\n")
print(filtered_text)

#####################################################################
# Stemming
# Reducing words to their base form
# Listening -> Listen, Jumped -> Jump

# NLTK has mutliple stemmers, we will use this one
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Stem word may end with 'i' instead of expected root form
# Use new example line
print("\n")
print(lineTwo)
tokenized_data = word_tokenize(lineTwo)
print(tokenized_data)

stemmed_text = [stemmer.stem(word) for word in tokenized_data]
print("\n")
print(stemmed_text)

#####################################################################
# Tagging
# Each word is made into a set, the word and a tag
# Tags represent as follows:
# JJ -> Adjectives
# NN -> Nouns
# RB -> Adverbs
# PRP -> Pronouns
# VB -> Verbs
#
# There are more tags than these examples
# Uncomment this line to see more information on tags
# nltk.help.upenn_tagset()

print("\n")
tagged_text = nltk.pos_tag(tokenized_data)
print(tagged_text)

#####################################################################
# Lemmatizing
# Lemmatizing is very similar to Stemming
# However, the resulting text will be a real English words
# Unlike when Stemming where results are in the form of:
# Challenging -> Challengi
#
# Lemming will give:
# Challenging -> Challenge
#
# Additionally, a 'lemma' is a word that represents a whole group
# of words, and that group of words is called a 'lexeme'
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Example
## This can also be done on a tokenized list of words
print("\n")
print("children")
print(lemmatizer.lemmatize("children"))

#####################################################################
# Chunking
# Chunking allows us to identify phrases
# A 'Phrase' is a group of words that work to serve a grammatical
# function
#
# Examples
# A Bear
# A Brown Bear
# A Googly-Eyed Brown Bear
#
# Uncomment line below if needed
# nltk.download("averaged_perception_tagger")

# Tokenize new example line
tokenized_data = makeTokens(lineThree)

# Then use Tagging from previous section
tagged_text = nltk.pos_tag(tokenized_data)
print(tagged_text)

# 'Chunk grammar' is a combination of rules on how a sentence
# should be 'chunked'
# Often with use of regular expressions
#
# Here, 'NP' means noun phrase.
# The rule we created is as follows:
# 1) Start with an optional [?] determiner (DT)
# 2) Can have any number [*] of adjectives (JJ)
# 3) End with a noun (NN)
chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"

# We need to create a 'chunk parser' to tell to use this grammar
chunk_parser = nltk.RegexpParser(chunk_grammar)

# The result will be in the form of a graph
res_tree = chunk_parser.parse(tagged_text)
res_tree.draw()

#####################################################################
# Chinking
# Chinking is used with Chunking
# But, Chinking is used to exclue a pattern
#
# The rule we created is as follows:
# Chunk: {<.*>+}
#        }<JJ>{
#
# 1) The first section: '{<.*>+}'
#        ({}) this tells the grammar to use/allow what rules are found
#        inside of these curly brackets
#    Here, <.*>+ tells us to include everything
#
# 2) The second section: '}<JJ>{'
#        (}{) this tells the grammar what to exclude
#    Here, <JJ> tells us to exclude adjectives

# We use triple quotes here because our grammar uses multiple lines
chinking_grammar = """
Chunk: {<.*>+}
       }<JJ>{"""

chunk_parser = nltk.RegexpParser(chinking_grammar)

res_tree = chunk_parser.parse(tagged_text)
res_tree.draw()

#####################################################################
# Named Entry Recognition (NER)
# Named Entities, are noun phrases that refer to specific
# locations, people, organizations, etc
#
# We can use NER to find Named Entities in our texts
# As well as determing which type they are
#
# Examples:
# PERSON -> Renton Thurston
# LOCATION -> Mumbo Mountain
# DATE -> April, 2001-04-01
#
# Note:
# There are more Named Entities than provided in this example

# Uncomment if needed
# nltk.download("maxent_ne_chunker")
# nltk.download("words")

res_tree = nltk.ne_chunk(tagged_text, binary=True)
res_tree.draw()

# We can use a simple function to get a list of all NE in a text
def getNamed(text):

    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    res_tree = nltk.ne_chunk(tags, binary=True)

    return set(
        "".join(items[0] for items in leaf)
        for leaf in res_tree
        if hasattr(leaf, "label") and leaf.label() == "NE"
    )

# Next example sentence
print(lineFour)
print(getNamed(lineFour))

#####################################################################
# Greater text analysis
#
# The previous examples have given us a number of tools to use for
# larger texts.
#
# Groups of texts are known as a 'Corpus'
# NLTK provies a number of Corpora for use when exploring how to
# further apply these tools
#
# Although this is a larger download and may take a moment...
# We can download these by uncommenting the following lines
# nltk.download("book")
# from nltk.book import *

# I will continue to use the quotes provided by Grunty
# from Banjo-Kazooie
with open('BK.txt') as f:
    quotes = f.read()


quotes_tokens = nltk.word_tokenize(quotes)
quotesCorpus = nltk.Text(quotes_tokens)
#####################################################################
# Concordance
#
# When we begin exploring a larger text we may want to see how often
# a word is used, with context!
#
# We can use a 'Concordance' to do so

print("\n")
quotesCorpus.concordance("Banjo")

print("\n")
quotesCorpus.concordance("Grunty")

#####################################################################
# Dispersion Plotting
#
# Using a Dispersion Plot, we can see how often a specific word
# appears, as well as where.
#
# This can also allow us to more easily compare the frequency of
# words with eachother

# Lets see how often the main antagonist talks about each of the
# main characters throughout the game
quotesCorpus.dispersion_plot(["Banjo", "bear","Kazooie", "bird", "Grunty", "Tooty"])

#####################################################################
# Frequency Distribution
#
# Here we can see which words are most frequent in Grunty's dialog
from nltk import FreqDist

freq_dist = FreqDist(quotesCorpus)

# A general overview of the results
print("\n")
print(freq_dist)

# Here are the 20 most common words in Grunty's dialog
print("\n")
print(freq_dist.most_common(20))

# Undoubtedly we can get much more information about this character
# If we filter out our stop words as we did earlier

grunty_talk = [
    word for word in quotesCorpus if word.casefold() not in stop_words]

# The results should be much more interesting than before
freq_dist = FreqDist(grunty_talk)
print("\n")
print(freq_dist.most_common(20))

# We can also graph this information
freq_dist.plot(20, cumulative=True)

#####################################################################
# Collocations
#
# A Collocation is simply a sequence of words that show up often
#
# Although, our example text may not contain many of these
# they can be useful in ensuring generated text does not feel
# stiff and robotic
print("\n")
quotesCorpus.collocations()

# Lemmatizing your text may reveal more collocations
lem_words = [lemmatizer.lemmatize(word) for word in quotesCorpus]
lemQuotesCorpus = nltk.Text(lem_words)

print("\n")
lemQuotesCorpus.collocations()
