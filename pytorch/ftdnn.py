import torch
import torchvision
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class FTDNN(nn.Module):
    def __init__(self,feature_len,device=torch.device('cuda')):
        super(FTDNN, self).__init__()
        self.device = torch.device(device)
        self.tdnn0_affine = nn.Conv1d(feature_len, 512, 5)
        self.tdnn0_relu = nn.ReLU()
        self.tdnn0_bn = nn.BatchNorm1d(512,affine=False)
        self.tdnn1_affine = nn.Conv1d(512,1024,1)
        self.tdnn1_relu = nn.ReLU()
        self.tdnn1_bn = nn.BatchNorm1d(1024)
        self.tdnn1_dropout = nn.Dropout()
        self.tdnn2_linear = nn.Conv1d(1024, 256, 2,dilation=2,bias=False)
        self.tdnn2_affine = nn.Conv1d(256, 1024, 2,dilation=2)
        self.tdnn2_relu = nn.ReLU()
        self.tdnn2_bn = nn.BatchNorm1d(1024)
        self.tdnn2_dropout = nn.Dropout()
        self.tdnn3_linear = nn.Conv1d(1024,256,1,bias=False)
        self.tdnn3_affine = nn.Conv1d(256,1024,1)
        self.tdnn3_relu = nn.ReLU()
        self.tdnn3_bn = nn.BatchNorm1d(1024)
        self.tdnn3_dropout = nn.Dropout()
        self.tdnn4_linear = nn.Conv1d(1024, 256, 2,dilation=3,bias=False)
        self.tdnn4_affine = nn.Conv1d(256, 1024, 2,dilation=3)
        self.tdnn4_relu = nn.ReLU()
        self.tdnn4_bn = nn.BatchNorm1d(1024)
        self.tdnn4_dropout = nn.Dropout()
        self.tdnn5_linear = nn.Conv1d(1024,256,1,bias=False)
        self.tdnn5_affine = nn.Conv1d(512,1024,1)
        self.tdnn5_relu = nn.ReLU()
        self.tdnn5_bn = nn.BatchNorm1d(1024)
        self.tdnn5_dropout = nn.Dropout()
        self.tdnn6_linear = nn.Conv1d(1024, 256, 2, dilation=3,bias=False)
        self.tdnn6_affine = nn.Conv1d(256, 1024, 2, dilation=3)
        self.tdnn6_relu = nn.ReLU()
        self.tdnn6_bn = nn.BatchNorm1d(1024)
        self.tdnn6_dropout = nn.Dropout()
        self.tdnn7_linear = nn.Conv1d(1024, 256, 2,dilation=3,bias=False)
        self.tdnn7_unfold2 = nn.Unfold((2, 256), dilation=(3, 1))
        self.tdnn7_affine = nn.Conv1d(1024,1024,1)
        self.tdnn7_relu = nn.ReLU()
        self.tdnn7_bn = nn.BatchNorm1d(1024)
        self.tdnn7_dropout = nn.Dropout()
        self.tdnn8_linear = nn.Conv1d(1024, 256, 2,dilation=3,bias=False)
        self.tdnn8_affine = nn.Conv1d(256, 1024, 2,dilation=3)
        self.tdnn8_relu = nn.ReLU()
        self.tdnn8_bn = nn.BatchNorm1d(1024)
        self.tdnn8_dropout = nn.Dropout()
        self.tdnn9_linear = nn.Conv1d(1024,256,1,bias=False)
        self.tdnn9_affine = nn.Conv1d(1024,1024,1)
        self.tdnn9_relu = nn.ReLU()
        self.tdnn9_bn = nn.BatchNorm1d(1024)
        self.tdnn9_dropout = nn.Dropout()
        self.tdnn10_affine = nn.Conv1d(1024,2048,1)
        self.tdnn10_relu = nn.ReLU()
        self.tdnn10_bn = nn.BatchNorm1d(2048)
        self.tdnn11_affine = nn.Linear(4096,512)
        self.tdnn11_relu = nn.ReLU()
        self.tdnn11_bn = nn.BatchNorm1d(512)
        self.tdnn12_affine = nn.Linear(512,512)
        self.tdnn12_relu = nn.ReLU()
        self.tdnn12_bn = nn.BatchNorm1d(512)
        self.tdnn13_affine = nn.Linear(512,30882)

    def forward(self,input):
        '''
        :param input: [batch, n_frame, n_feature]
        :return:[batch, output_features]
        '''
        #[batch_size, seq_len, max_word_len] = input.size()
        #input = input.unsqueeze(1)
        input = input.transpose(2, 1)
        output = self.tdnn0_affine(input)
        output = self.tdnn0_relu(output)
        output = self.tdnn0_bn(output)
        output = self.tdnn1_affine(output)
        output = self.tdnn1_relu(output)
        output = self.tdnn1_bn(output)
        output = self.tdnn1_dropout(output)
        tdnn2l = self.tdnn2_linear(output)
        output = self.tdnn2_affine(tdnn2l)
        output = self.tdnn2_relu(output)
        output = self.tdnn2_bn(output)
        output = self.tdnn2_dropout(output)
        tdnn3l = self.tdnn3_linear(output)
        output = self.tdnn3_affine(tdnn3l)
        output = self.tdnn3_relu(output)
        output = self.tdnn3_bn(output)
        output = self.tdnn3_dropout(output)
        tdnn4l = self.tdnn4_linear(output)
        output = self.tdnn4_affine(tdnn4l)
        output = self.tdnn4_relu(output)
        output = self.tdnn4_bn(output)
        output = self.tdnn4_dropout(output)
        tdnn5l = self.tdnn5_linear(output)
        output = torch.cat([tdnn5l,tdnn3l[:,:,3:-3]],dim=1)    #连接的时候需要对tdnn3l进行截断
        output = self.tdnn5_affine(output)
        output = self.tdnn5_relu(output)
        output = self.tdnn5_bn(output)
        output = self.tdnn5_dropout(output)
        tdnn6l = self.tdnn6_linear(output)
        output = self.tdnn6_affine(tdnn6l)
        output = self.tdnn6_relu(output)
        output = self.tdnn6_bn(output)
        output = self.tdnn6_dropout(output)
        tdnn7l = self.tdnn7_linear(output)
        output = self.tdnn7_unfold2(tdnn7l.transpose(2,1).unsqueeze(1))
        output = torch.cat([output,tdnn4l[:,:,6:-9],tdnn2l[:,:,9:-11]],dim=1)  #连接的时候需要对tdnn4l和tdnn2l进行截断
        output = self.tdnn7_affine(output)
        output = self.tdnn7_relu(output)
        output = self.tdnn7_bn(output)
        output = self.tdnn7_dropout(output)
        tdnn8l = self.tdnn8_linear(output)
        output = self.tdnn8_affine(tdnn8l)
        output = self.tdnn8_relu(output)
        output = self.tdnn8_bn(output)
        output = self.tdnn8_dropout(output)
        tdnn9l = self.tdnn9_linear(output)
        output = torch.cat([tdnn9l,tdnn8l[:,:,:-3],tdnn6l[:,:,6:-9],tdnn4l[:,:,9:-12]],dim=1)
        output = self.tdnn9_affine(output)
        output = self.tdnn9_relu(output)
        output = self.tdnn9_bn(output)
        output = self.tdnn9_dropout(output)
        output = self.tdnn10_affine(output)
        output = self.tdnn10_relu(output)
        output = self.tdnn10_bn(output)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        output = torch.cat((mean, std), dim=1)
        output = self.tdnn11_affine(output)        # 特征输出层
        output = self.tdnn11_relu(output)
        output = self.tdnn11_bn(output)
        output = self.tdnn12_affine(output)
        output = self.tdnn12_relu(output)
        output = self.tdnn12_bn(output)
        output = self.tdnn13_affine(output) #分类使用
        output = F.softmax(output, dim=1)
        return output





feature_len = 23
batch_size = 1
time_square = 513
input = torch.randn(batch_size, time_square,feature_len ).to('cuda')
net = FTDNN(feature_len).to('cuda')
net.eval()
output = net(input)
print(output.shape)
# for key,value in net.state_dict().items():
#     print(key,value.size())
# torch.save(net.state_dict(), "ftdnn.pt")
# net.load_state_dict(torch.load('ftdnn.pt',map_location="cuda:0"))
# output = net(input)
# writer = SummaryWriter('tensorboard/xvector-kaldi')
# writer.add_graph(net, input)   #graph页
#
# #script model save 1
# sm = torch.jit.script(net)
# sm.eval()
# sm.save("ftdnn_s1.pt")
# output = sm.forward(input)
#
#
# #script model save 2
# traced_script_module = torch.jit.trace(net, input)
# traced_script_module.eval()
# traced_script_module.save("ftdnn_s2.pt")
#
# from torchsummary import summary
# summary(net,(time_square,feature_len))
# print(net.state_dict().keys())
# print(net.state_dict()["bn1.num_batches_tracked"].shape)

torch.onnx.export(net,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  "tdnn-f.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size',1:'time_square'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

