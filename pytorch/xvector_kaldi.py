import torch
import torchvision
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TDNN(nn.Module):
    def __init__(self,feature_len,device=torch.device('cuda')):
        super(TDNN, self).__init__()
        self.device = torch.device(device)
        self.unfold1 = nn.Unfold((5,20))  #使用unfold取连续5帧特征
        self.h1 = nn.Conv1d(100,512,1)
        self.bn1 = nn.BatchNorm1d(512,affine=False)
        self.unfold2 = nn.Unfold((3, 512),dilation=(2,1))  # 使用unfold 隔1帧抽，抽3帧特征
        self.h2 = nn.Conv1d(1536,512,1)
        self.bn2 = nn.BatchNorm1d(512)
        self.unfold3 = nn.Unfold((3, 512), dilation=(3, 1))  # 使用unfold 隔2帧抽，抽3帧特征
        self.h3 = nn.Conv1d(1536, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.h4 = nn.Conv1d(512, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.h5 = nn.Conv1d(512, 1500, 1)
        self.bn5 = nn.BatchNorm1d(1500)
        self.h6 = nn.Linear(3000,512)
        self.bn6 = nn.BatchNorm1d(512)
        self.h7 = nn.Linear(512,512)
        self.bn7 = nn.BatchNorm1d(512)
        self.h8 = nn.Linear(512,10000)


    def forward(self,input):
        '''
        :param input: [batch, n_frame, n_feature]
        :return:[batch, output_features]
        '''
        #[batch_size, seq_len, max_word_len] = input.size()
        input = input.unsqueeze(1)
        output = self.unfold1(input).double()
        output = self.h1(output)
        output = F.relu(output)
        output = self.bn1(output)
        output = self.unfold2(output.transpose(2,1).unsqueeze(1))
        output = self.h2(output)
        output = F.relu(output)
        output = self.bn2(output)
        output = self.unfold3(output.transpose(2,1).unsqueeze(1))
        output = self.h3(output)
        output = F.relu(output)
        output = self.bn3(output)
        output = self.h4(output)
        output = F.relu(output)
        output = self.bn4(output)
        output = self.h5(output)
        output = F.relu(output)
        output = self.bn5(output)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        output = torch.cat((mean, std), dim=1)
        output = self.h6(output)    # 特征输出层
        output = F.relu(output)
        output = self.bn6(output)
        output = self.h7(output)
        output = F.relu(output)
        output = self.bn7(output)
        output = self.h8(output)   # 分类使用
        output = F.softmax(output,dim=1)
        return output





feature_len = 20
batch_size = 10
time_square = 513
input = torch.randn(batch_size, time_square,feature_len ).type(torch.float64).to('cuda')
net = TDNN(feature_len).type(torch.float64).to('cuda')
net.eval()
torch.save(net.state_dict(), "xvector.pt")
net.load_state_dict(torch.load('xvector.pt',map_location="cuda:0"))
output = net(input)
writer = SummaryWriter('tensorboard/xvector-kaldi')
writer.add_graph(net, input)   #graph页

#script model save 1
sm = torch.jit.script(net)
sm.eval()
sm.save("xvector_s1.pt")
output = sm.forward(input)


#script model save 2
traced_script_module = torch.jit.trace(net, input)
traced_script_module.eval()
traced_script_module.save("xvector_s2.pt")

from torchsummary import summary
summary(net,(time_square,feature_len))
print(net.state_dict().keys())
print(net.state_dict()["bn1.num_batches_tracked"].shape)


