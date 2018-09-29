import torch
import torch.nn as nn
import random

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 70

device = torch.device("cuda:1" if torch.cuda.is_available() else"cpu")
torch.backends.cudnn.enabled = False

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1:"EOS", 2:"UNK"}
        self.n_words = 3

    def addSentence(self, sentence): #文を空白で区切って一つずつ単語を辞書にしていく
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word): #wordはtxt
        #単語を追加して辞書を作成
        if word not in self.word2index: #単語が辞書になければ追加
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1 #単語数

        else:
            self.word2count[word] += 1 #辞書にあった場合countを1増やす

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unidodedata.category(s) != 'Mn'
        )

    def normalizeString(self,s):
        return s.strip()

def readLangs(lang1,lang2,domain,reverse=False): 
    #lang1, lang2は発話者の名前('hu','ro)(string)
    #domainはドメイン名('cleaning')(string)
    print("Reading lines...")

    #Read the file and split into lines
    lines = open("data/%s/%s-%s.txt" % (domain,lang1,lang2), encoding='utf-8').read().strip().split('\n')

    #Split every line into pairs and normalize
    #発話対を入力文と出力分に分ける
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
def filterPair(p): #発話対内の単語数がMAX_LENGTHより少なければ値を返す
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):#FilterPairの条件を満たしている発話対のみ配列に入れて返す
    return [pair for pair in pairs if filterPair(pair)]

#Prepairing Training Data
def tensor_indexFromSentence(lang, sentence):
    #langは参照するLangクラス,sentenceはid化したい文(形態素解析済み, 単語間は空白区切り),
    #sentenceをid化した配列のtensor(size[len(words),1])を返す
    ids = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            ids.append(lang.word2index[word])
        else :
            ids.append(UNK_token)
    #文の最後に<EOS>のidを追加
    ids.append(EOS_token)
    return torch.tensor(ids, dtype=torch.long, device=device)

#発話対を配列にした配列を受け取って、id化したtensorの配列を返す
def tensor_FromPair(input_lang, output_lang, pair):
    input_tensor = tensor_indexFromSentence(input_lang, pair[0])
    target_tensor = tensor_indexFromSentence(output_lang, pair[1])
    return(input_tensor, target_tensor)


#Load Data Corpus #入力文用単語辞書と出力文用単語辞書を作成する, 加えて発話対を発話応答に分けた配列を返す
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

###Emotionの辞書作成
#Emotionの種類は5種類設定、そのまま記述
emo2index = {"平静":0,"怒り":1,"悲しみ":2, "喜び":3, "安心":4}
index2emo = {"平静","怒り","悲しみ","喜び","安心"}

#Load Robot_Emotion 
def LoadEmo(domain,lang3):
    #lang3にはemotion の主を入れる( robot or human)
    emotions = open('data/%s/%s_emotions.txt' % (domain,lang3), encoding='utf-8').read().strip().split('\n')
    emo_id = [[emo2index[e]] for e in emotions]
    emo_tensor = torch.tensor(emo_id, dtype=torch.long, device=device)
    print("Read %s %s-emotions" % (len(emotions),lang3))
    return emo_tensor

