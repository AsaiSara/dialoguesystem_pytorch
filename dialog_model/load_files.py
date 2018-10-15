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
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS", 3:"UNK"}
        self.n_words = 4

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
def indexFromSentence(lang, sentence):
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
    length = len(ids)
    ids += [0]*(MAX_LENGTH-len(ids))
    return ids, length

def tensor_indexFromSentence(lang,sentence):
    ids,_ = indexFromSentence(lang,sentence)
    return torch.tensor(ids,dtype=torch.long, device=device).view(-1,1) 


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
    #emo_tensor = torch.tensor(emo_id, dtype=torch.long, device=device)
    print("Read %s %s-emotions" % (len(emotions),lang3))
    return emo_id

#training用のデータ準備
#randomに学習データのペアと感情を選んでn_iter回数文の配列を作成
#mini-batch処理を追加

def training_set_emo(input_lang, output_lang, pairs, emo_id, n_iters):
    training_sets = [random.choice(list(zip(pairs, emo_id))) for i in range(n_iters)]
    training_pairs = [tensor_FromPair(input_lang, output_lang, pair) for pair, _ in training_sets]
    training_emotions = [torch.tensor(emo,dtype=torch.long,device=device) for _, emo in training_sets]
    return training_pairs, training_emotions

batch_size = 30
def generate_batch(input_lang, output_lang, pairs, batch_size, shuffle=True):
    random.shuffle(pairs)
    
    for i in range(len(pairs)//batch_size):
        batch_pairs = pairs[batch_size* i:batch_size * (i+1)]

        input_batch = []
        target_batch = []
        input_lens = []
        target_lens = []
        for input_seq, target_seq in batch_pairs:
            #文中の単語のidの配列を受け取る
            input_seq, input_len = indexFromPair(input_lang,input_seq)
            target_seq, target_len = indexFromPair(output_lang,output_seq)
            #id配列をbatch用配列に追加、単語数も長さの配列に追加
            input_batch.append(input_seq)
            target_batch.append(target_seq)
            input_lens.append(input_len)
            target_lens.append(target_len)
        #配列をtensorにする
        input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
        target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
        input_lens = torch.tensor(input_lens)
        target_lens = torch.tensor(target_lens)

        # sort 
        #inputの単語数を大きいものからsortした配列と、そのidを記録
        input_lens, sorted_idxs = input_lens.sort(0, descending=True)
        #単語数の多い文から文の順をsortし、転置する(まとまりを文ごから単語の出現順に変える)
        input_batch = input_batch[sorted_idxs].transpose(0,1)
        #すべてPADDINGになっている要素を省く
        input_batch = input_batch[:input_lens.max().item()]

        #input_batchと順番をそろえて、同様に転置する
        target_batch = target_batch[sorted_idx].transpose(0,1)
        target_batch = target_batch[:target_lens.max().item()]
        target_lens = target_lens[sorted_idxs]

        yield input_batch, input_lens, target_batch, target_lens
            





