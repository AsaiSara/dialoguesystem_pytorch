SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1:"EOS", 2:"UNK"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unidodedata.category(s) != 'Mn'

    def 
            
 
   def readLangs(lang1,lang2,reverse=False):
        print("Reading lines...")

        #Read the file and split into lines
        lines = open("cleaning/%s-%s.txt" % (lang1,lang2), encoding='utf-8').read().strip().split('\n')

        #Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

        #Reverse pairs, make Lnaf instance
        if reverse      
