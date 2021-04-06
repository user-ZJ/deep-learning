import torch
import torchvision
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TDNN(nn.Module):
    def __init__(self,feature_len):
        super(TDNN, self).__init__()
        self.h1 = nn.Conv1d(feature_len,512,5)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512,affine=False)
        self.h2 = nn.Conv1d(512,512,3,dilation=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.h3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.h4 = nn.Conv1d(512, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.h5 = nn.Conv1d(512, 1500, 1)
        self.bn5 = nn.BatchNorm1d(1500)
        self.h6 = nn.Linear(3000,512)
        self.bn6 = nn.BatchNorm1d(512)
        self.h7 = nn.Linear(512,512)
        self.bn7 = nn.BatchNorm1d(512)
        self.h8 = nn.Linear(512, 10000)


    def forward(self,input):
        '''
        :param input: [batch, n_frame, n_feature]
        :return:[batch, new_seq_len, output_features]
        '''
        #[batch_size, seq_len, max_word_len] = input.size()
        input = input.transpose(2,1)
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
        output = torch.cat((mean, std), dim=1)  #stats_pooling
        output = self.h6(output)    #xvector特征输出
        output = F.relu6(output)
        output = self.bn6(output)
        output = F.relu6(self.h7(output))
        output = self.bn7(output)
        output = self.h8(output)  # 分类使用
        output = F.softmax(output,dim=1)
        return output



feature_len = 20
batch_size = 1
time_square = 513
input = torch.randn(batch_size, time_square,feature_len).to('cuda')
net = TDNN(feature_len).to('cuda')
net.eval()
torch.save(net.state_dict(), "xvector.pt")
net.load_state_dict(torch.load('xvector.pt',map_location="cuda:0"))
output = net(input)
print(input.size(),output.size())

#script model save 1
sm = torch.jit.script(net)
sm.eval()
sm.save("xvector_s1.pt")

#script model save 2
traced_script_module = torch.jit.trace(net, input)
traced_script_module.eval()
traced_script_module.save("xvector_s2.pt")

torch.onnx.export(net,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  "xvector.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size',1:'time_square'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})



