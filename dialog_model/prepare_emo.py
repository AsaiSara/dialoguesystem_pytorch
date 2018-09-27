######## 他のファイルに移したので使わない######


emo2index = {"平静":0,"怒り":1,"悲しみ":2, "喜び":3, "安心":4}
index2emo = {"平静","怒り","悲しみ","喜び","安心"}

#ロボット側の感情ラベルの読み込み
emotions = open('cleaning/robot_emotions.txt', encoding='utf-8').read().strip().split('\n')
emo_index = [[emo2index[e]] for e in emotions]
emo_tensor = torch.tensor(emo_index, dtype=torch.long, device=device)

#下のクラスは今は使わない、入力から感情を予測する場合に使う
class PredictEmotion(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(PredictEmotion, self).__init__()
        self.li1 = nn.Linear(hidden_size, hidden_size)
        self.li2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.li1(x)
        h = F.tanh(h)
        y = self.li2(h)
        y = F.log_softmax(y,dim=1)
        return y
