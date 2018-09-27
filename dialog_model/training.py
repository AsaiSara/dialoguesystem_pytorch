import time
import math
teacher_forcing_ratio = 0.5


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


def train(input_tensor, target_tensor, emo_id, encoder, decoder, encoder_optimizer, decoder_optimizer, emotion_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    emotion_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensore[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        #Teacher forcing:Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, emo_id, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_tensor[di])
            decoder_input = target_tensor[di] #Teacher forcing

        else:
            #Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output decoder_hidden, decoder_attention = decoder(
                    decoder_input, emo_id, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  #detach from history as input

                loss += criterion(decoder_output[0],target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        emotion_optimizer.step()

        return loss.item() / target_length 


#robo_emo_tensor追加
def trainIters(encoder, decoder, robo_emo_tensor, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    #encoder: EncoderRNN(input,hidden)クラス, decoder: AttenDecoderRNN(hidden,output)クラス
    #######robo_emo_tensorでtensorのrobo_emoを渡して使いたいが、未完成
    start = time.time()
    plot_losses = []
    print_loss_total = 0  #Reset every print_every
    print_emo_loss_total = 0
    plot_loss_total = 0 #Reset every plot_every

    #oprimizer : SGDを使用
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    emotion_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    #randomに学習データを選んでn_ters回数分の配列を作成、感情は分けて配列を作成 
    training_sets = [random.choice(list(zip(pairs, robo_emo_tensor))) for i in range(n_iters)]
    training_pairs = [tensorsFromPair(pair) for pair, _ in training_sets]
    training_emotions = [emo for _, emo in training_sets]
    #######上のデータ処理をtrainの外でやりたい

    #損失関数：NLLoss(CrossEntropyLossでsoftmaxを噛ませない損失関数)
    criterion = nn.NLLoss()
    #発話対をinputとtargetに分けて変数に入れて、train関数に入れて損失関数を出す
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter -1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        emo_id = training_emotions[iter - 1]
　　　　#損失関数をtrain関数で算出する
        loss = train(input_tensor, target_tensor, emo_id, encoder, decoder, encoder_optimizer, decoder_optimizer, emotion_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss
　　　　#print_every回数ごとに開始からの時間と、
　　　　#トレイン回数と学習の終わった割合とprint_everyごとの平均ロスを出力
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

