import torch
import torch.nn as nn

device = torch.device("cuda:1" if torch.cuda.is_available() else"cpu")

#Encoder model
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def forward(self, input, hidden):
        embeded = self.embedding(input).view(1,1,-1)
        output = embeded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size, device = device)

class DecoderRNN(nn.Module):
    def __init__(self,hidden_size, output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size, device=device)

class AttenDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.emotion_embedding = nn.Embedding(5,5)
        self.attn = nn.Linear(self.hidden_size * 2 + 5, self.max_length)
        self.attn_combine = nn.Linear(self.hiddensize * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size + 5, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, emotion, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1,1,self.hidden_size)
        embeded = self.dropout(embedded)

        emotion_emb = self.emotion_embedding(emotion).view(-1,1,5)

        ht, hidden = self.gru(torch.cat((embeded, emotion_emb),dim=2),hidden)

        score = torch.bmm(encoder_outputs.unsqueeze(0), torch.transpose(ht,1,2))
        attn_weights = F.softmax(score, dim=1)
        ct = torch.bmm(torch.transpose(attn_weights, 1,2),encoder_outputs.unsqueeze(0))
        ht_var = torch.cat((ct, ht),dim=2)
        ht_var = F.tanh(ht_var)
        ht_var = self.out(ht_var)
        output = F.log_softmax(ht_var, dim=2)

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)


