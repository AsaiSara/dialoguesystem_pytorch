import time
import math

from load_files import * 
from model import *
from torch import optim

#teacher_forcingを使う割合、使わない場合はdecoder_outputの最も確率が高いidを次の入力にする
teacher_forcing_ratio = 0.5

#時間の出力のための関数
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

def timeSince(since,percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es -s
    return '%s (- %s)' %(asMinutes(s), asMinutes(rs))

#Training 
def train(input_tensor, target_tensor, emo_id, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #input_tensor,target_tensore: 発話文と応答文をid化した配列のtensor(1文の単語配列)
    #emo_id: emotion id の配列のtensor(発話対1ペアに対する感情ラベル)
    #encoder: EncoderRNNのmodel, decoder: decoderRNNのmoddel 
    #criterion: loss function

    #encoder_hidden, optimizerを初期化
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #input_tensor,target_tensorの長さを変数に入れる
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    #encoder_outputs を0行列のtensorにしておく
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    #loss の初期化
    loss = 0

    #inputのtensorを一つずつ(1単語ずつ)取り出してencodeする
    #出力はattentionで使うのでencoder_outputsのtensorに入れていく
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #encoder_outputは3次元なので3次元の値のみ取り出して, encoder_outputsのtensor配列に入れる
        encoder_outputs[ei] = encoder_output[0,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        #Teacher forcing:Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, emo_id, decoder_hidden, encoder_outputs)
            print(decoder_output[0].size(),target_tensor[di].size())
            loss += criterion(decoder_output[0], target_tensor[di])
            #decoderの入力に一つ前の正解データ(target)をそのまま次のdecoder_inputで使う(Teacher Forcing)
            decoder_input = target_tensor[di] #Teacher forcing

    else:
        #Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, emo_id, decoder_hidden, encoder_outputs)
            #最大値とその配列番号を渡す、softmaxで最大の値を次のdecoder_inputで使う
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  #detach from history as input
            #損失関数に正解データとoutputの値とのNLLossを加える。
            loss += criterion(decoder_output[0],target_tensor[di])
            #decoder_inputの値がEOSのidであった場合ループを抜ける(target_lengthより前にEOSが出た場合)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length 


#emo_tensor追加
def trainIters(encoder, decoder, emo_id, n_iters, input_lang, output_lang, pairs, print_every=1000, plot_every=100, learning_rate=0.01):
    #encoder: EncoderRNN(input,hidden)クラス, decoder: AttenDecoderRNN(hidden,output)クラス
    #n_iters: 学習させる回数(int)
    #emo_tensor: emotionのidをtensor
    start = time.time()
    plot_losses = []
    print_loss_total = 0  #Reset every print_every
    print_emo_loss_total = 0
    plot_loss_total = 0 #Reset every plot_every

    #oprimizer : SGDを使用
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    #randomに学習データを選んでn_ters回数分の配列を作成、感情は分けて配列を作成 
    #training_sets = [random.choice(list(zip(pairs, emo_tensor))) for i in range(n_iters)]
    #training_pairs = [tensor_FromPair(input_lang, output_lang, pair) for pair, _ in training_sets]
    #training_emotions = [emo for _, emo in training_sets]
    #######上のデータ処理をtrainの外でやりたい

    #Load_filesの関数を使ってtraining用のランダムにn_iter回数文のtensor配列をつくる 
    training_pairs, training_emotions = training_set_emo(input_lang, output_lang, pairs, emo_id,n_iters)

    #損失関数：NLLLoss(CrossEntropyLossでsoftmaxを噛ませない損失関数)
    criterion = nn.NLLLoss()
    #発話対をinputとtargetに分けて変数に入れて、train関数に入れて損失関数を出す
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter -1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        emo_id = training_emotions[iter - 1]
        #損失関数をtrain関数で算出する
        loss = train(input_tensor, target_tensor, emo_id, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss
        #print_every回数ごとに開始からの時間と、
        #学習回数, 学習の終わった割合, print_everyごとのLossの平均を出力
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s ( %d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        #matplotlibで損失関数の変化をplot
        if iter % print_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

#Plotting results
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    #this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

###Prepare Datas###
from prepare_data import SetData
input_lang, output_lang, train_pairs, dev_pairs, test_pairs, emo_id = SetData('hu','ro','cleaning','robot',200,1000)

###Training###
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
trainIters(encoder1, attn_decoder1, emo_id, 100, input_lang, output_lang, train_pairs, print_every=10)


