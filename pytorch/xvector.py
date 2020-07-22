from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from time import time

import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TDNN(nn.Module):
    def __init__(self,feature_len):
        super(TDNN, self).__init__()
        self.h1 = nn.Conv1d(feature_len,512,5)
        self.bn1 = nn.BatchNorm1d(512)
        self.h2 = nn.Conv1d(512,512,3,dilation=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.h3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.h4 = nn.Conv1d(512, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.h5 = nn.Conv1d(512, 1500, 1)
        self.bn5 = nn.BatchNorm1d(1500)
        self.line1 = nn.Linear(3000,512)
        self.bn6 = nn.BatchNorm1d(512)
        self.line2 = nn.Linear(512,512)
        self.bn7 = nn.BatchNorm1d(512)


    def forward(self,input):
        '''
        :param input: [batch, n_frame, n_feature]
        :return:[batch, new_seq_len, output_features]
        '''
        #[batch_size, seq_len, max_word_len] = input.size()

        output = self.h1(input)
        output = F.relu6(output)
        output = self.bn1(output)
        output = F.relu6(self.h2(output))
        output = self.bn2(output)
        output = F.relu6(self.h3(output))
        output = self.bn3(output)
        output = F.relu6(self.h4(output))
        output = self.bn4(output)
        output = F.relu6(self.h5(output))
        output = self.bn5(output)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        output = torch.cat((mean, std), dim=1)
        output = F.relu6(self.line1(output))
        output = self.bn6(output)
        output = F.relu6(self.line2(output))
        output = self.bn7(output)
        return output



feature_len = 20
batch_size = 10
input = torch.randn(batch_size, feature_len, 2500).to('cuda')
net = TDNN(feature_len).to('cuda')
net.eval()
torch.save(net.state_dict(), "xvector.pt")
net.load_state_dict(torch.load('xvector.pt'))
output = net(input)

#script model save 1
sm = torch.jit.script(net)
sm.eval()
sm.save("xvector_s1.pt")

#script model save 2
traced_script_module = torch.jit.trace(net, input)
traced_script_module.eval()
traced_script_module.save("xvector_s2.pt")

from torchsummary import summary
summary(net,(feature_len,2500))


