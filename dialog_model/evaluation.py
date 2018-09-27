def evaluate(encoder, decoder, sentence, emo_id, max_length=MAX_LENGTH):
#encoderは学習後EncoderRNNモデル, decoderは学習後AttnDecodeRNNモデル, sentenceは入力文
#emo_idはシステム側の指定した感情のid(数値0~4いずれか), max_lengthでdecoder_attentionsの長さを決める
    with torch.no_grad():
        #テストデータのinputのデータクラス(input_lang)をもとに、
        #sentenceをid化してtensorにしたものをinput_tensorに入れている
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        #encoderクラスのHiddenサイズの0行列のtensorをencoder_fiddenに入れる
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            #input_tensorのei番目の単語と前のencoderのhidden stateをencodeしてoutput とhiddenを出す
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            #encoderのoutputの配列を作る
            encoder_outputs[ei] += encoder_output[0,0]

            #decoderの入力は入れるものがないから？SOSのidのtensorを入れる
            decoder_input = torch.tensor([[SOS_token]], device=device) #SOS
            #decoderのhiddenは学習時と同じくencoderのhidden stateを入れる
            decoder_hidden = encoder_hidden
            
            #emo_id をtensorにしておく
            emo_id = torch.tensor(emo_id, dtype=torch.long, device=device)
            
            decoded_words = []
            #max_length*maxlengthのゼロ行列のtensorをdecoderのアテンションとして渡す
            decoder_attentions = torch.zeros(max_length, max_length)

            #max_lengthの分の長さを一つずつdecodeする
            for di in range(max_length):  
                #上で設定したdecoderのinputとhidden とemo_id とencoder_outputsをでコードする
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, emo_id, decoder_hidden, encoder_outputs)
                #tensorであるdecoder outputの値の最大値とその値の配列の番号(引数は最大値の配列の大きさ)
                topv, topi = decoder_output.data.topk(1)

                #decodeで出した値がEOS のidのtensorであった場合、wordに<EOS>を追加してループを抜ける
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:　#decodeしたoutputを単語に変換して、wordとして加える
                    decoded_words.append(output_lang.index2word[topi.item()])
                    #decodeのinputとして
                    decoder_input = topi.squeeze().detach()

                return decoded_words, decoder_attentions[:di + 1]

