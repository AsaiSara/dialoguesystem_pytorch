SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 70

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

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unidodedata.category(s) != 'Mn'
        )

    def normalizeString(self,s):
        return s.strip()


def readLangs(lang1,lang2,reverse=False):
    print("Reading lines...")

    #Read the file and split into lines
    lines = open("data/cleaning/%s-%s.txt" % (lang1,lang2), encoding='utf-8').read().strip().split('\n')

    #Split every line into pairs and normalize
    pairs = [[ s.strip() for s in l.split('\t')] for l in lines]

    #Reverse pairs, make Lnaf instance
    if reverse:
        pairs = [list(reverse(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

#Filtering Data    
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#Prepairing Training Data
def indexFromSentence(lang, sentence):
    ids = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            ids.append(lang.word2index[word])
        else :
            ids.append(UNK_token)
        return ids

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSetence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return(input_tensor, target_tensor)


#Load Data Corpus 
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Couonted words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pair


  
  
  
  
  
  
  
  
  
  
  
  

