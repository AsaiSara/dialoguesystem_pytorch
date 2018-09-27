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

def readLangs(lang1,lang2,domain,reverse=False):
    print("Reading lines...")

    #Read the file and split into lines
    lines = open("data/%s/%s-%s.txt" % (domain,lang1,lang2), encoding='utf-8').read().strip().split('\n')

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
def prepareData(lang1, lang2, domain , reverse=False):
    #reverseにすると入力文と出力文が逆になる
    #lang1はinput文の名前,lang2はoutput文の名前,domainは対話のドメイン名
    #lang1 = 'hu' #human
    #lang2 = 'ro' #robot
    #domain = 'cleaning' #file読み込み時のdirectry指定に使うだけ
    #input文のLangクラス,output文のLangクラス, 発話対の配列を返す
    #この時点ではクラスに名前とreverseの有無のみ入っている
    input_lang, output_lang, pairs = readLangs(lang1, lang2, domain, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #発話対のデータをフィルターにかける
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Couonted words...")
    #発話対に含まれる単語をクラス内の辞書に追加していく
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    #input文のクラス(辞書),output文のクラス(辞書)を返す
    return input_lang, output_lang, pair


class Emo(self, name):
    self.name = name
    self.emo2index = {"平静":0,"怒り":1,"悲しみ":2, 
                        "喜び":3, "安心":4}
    self.index2emo = {"平静","怒り","悲しみ","喜び","安心"}

#Load Robot_Emotion 
def LoadEmo(domain,lang3):
    #lang3= robot or human
    emo = Emo(lang3)
    emotions = open('%s/%s_emotions.txt' % (domain,lang3), encoding='utf-8').
        read().strip()split('\n')
    emo_id = [[emo.emo2index[e] for e in emotions]
    emo_tensor = torch.tensore(emo_id, dtype=torch.long, device=decice)
    print("Read %s %s-emotions" % len(emotions))
    return emo_tensor

