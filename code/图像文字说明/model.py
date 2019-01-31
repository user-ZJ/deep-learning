import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # hidden_size 隐藏层维度
        self.embed_size = embed_size  # embed_size 输入tensor大小
        self.vocab_size = vocab_size # vocab_size 词汇量的大小
        self.have_gpu=torch.cuda.is_available()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,dropout=0.5,batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size) # vocab_size 词汇量的大小
        self.word_embeddings=nn.Embedding(9955,embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features 输入
        # captions 隐藏层状态
        batch_size = features.size()[0]
        self.hidden = self.init_hidden(batch_size)
        word_count = captions.size()[1]
        features = features.unsqueeze(1)
        embeds = self.word_embeddings(captions[:,:word_count-1])
        inputs = torch.cat((features,embeds),1)
        #print(inputs.shape)
        x, self.hidden = self.lstm(inputs, self.hidden)
        x = self.dropout(x)
        x = x.view(x.size()[0], x.size()[1], self.hidden_size)
        x = self.fc(x)
        x = F.log_softmax(x,dim=2)
        return x


    def init_hidden(self,batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        # return (torch.zeros(num_layers, batch_size, self.hidden_dim),
        #         torch.zeros(num_layers, batch_size, self.hidden_dim))
        weight = next(self.parameters()).data
        if self.have_gpu:
            return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to('cuda'),
                    weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to('cuda'))
        else:
            return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                    weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        results = []
        x = inputs
        if states is None:
            states = self.init_hidden(1)
        for i in range(max_len):
            x,states=self.lstm(x,states)
            x = self.fc(x)
            x = x.view(1,1,-1)
            wordindex = torch.argmax(x.squeeze()).tolist()
            results.append(wordindex)
            if wordindex==1:
                return results
            x = self.word_embeddings(torch.argmax(x,dim=2))
        return results



        # results = []
        # hidden = self.init_hidden(1)
        # x ,hidden = self.lstm(inputs,hidden)
        # x = x.view(x.size()[0], x.size()[1], self.hidden_size)
        # x = self.fc(x)
        # x = F.softmax(x,dim=2)
        # print(x)
        # x = torch.argmax(x,dim=2)
        # print(x)
        # print(x.squeeze().cpu().numpy().tolist())
        # results.append(x.squeeze().cpu().numpy().tolist())
        # print(results)
        #
        # for i in range(5):
        #     x = self.word_embeddings(x)
        #     x, hidden = self.lstm(inputs, hidden)
        #     x = x.view(x.size()[0], x.size()[1], self.hidden_size)
        #     x = self.fc(x)
        #     x = F.softmax(x, dim=2)
        #     x = torch.argmax(x, dim=2)
        #     results.append(x.squeeze().cpu().numpy().tolist())