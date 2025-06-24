import Stemmer

class Tokenizer:
    def __init__(self):
        self.stemmer = Stemmer.Stemmer('english') 

    # Runtime complexity: O(n), where n is the number of characters in the file
    # Taking each character is the file, checking if it's alphanumeric and building a token
    def tokenize(self, text):
        tokens = []
        currToken = ""
        for c in text:
            if c.isascii() and c.isalnum():
                currToken += c
            else:
                if currToken:
                    stemmed = self.stemmer.stemWord(currToken.lower())
                    tokens.append(stemmed)
                    currToken = ""
        if currToken:
            tokens.append(currToken.lower())
        return tokens

    # Runtime complexity: O(m), where m is the number of tokens.
    # Adding each token and calculating its frequency by creating a dictionary
    def computeWordFrequencies(self, tokens):
        tokenFreq = {}
        for t in tokens:
            if t not in tokenFreq:
                tokenFreq[t] = 1
            else:
                tokenFreq[t] += 1
        return tokenFreq
    
    def compute_ngrams(self, tokens, n):
        return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
