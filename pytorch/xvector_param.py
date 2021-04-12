import os
import re
import numpy as np
import pickle
import torch

def subString1(template):
    rule = r'<(.*?)>'
    slotList = re.findall(rule,template)
    return slotList

def subString2(template):
    rule = r'>(.*?)<'
    slotList = re.findall(rule,template)
    return slotList

def subString3(template):
    rule = r'\[(.*?)\]'
    slotList = re.findall(rule,template)
    return slotList

def subString4(template):
    rule = r'\](.*?)\['
    slotList = re.findall(rule,template)
    return slotList

f = open("xvector.txt","rt")
data = f.read()
data = data.replace('\n','|')
keys = subString1(data)
print(keys)
values = subString2(data)
print("values length:",len(values))
matrixs = subString3(data)
print("matrixs length",len(matrixs))
param_list = []
for matrix in matrixs:
    params = []
    rows = matrix.split('|')
    for row in rows:
        sub_param = None
        indexs = row.split()
        if(len(indexs)>0):
            sub_param = []
            for index in indexs:
                sub_param.append(float(index))
        if sub_param is not None:
            params.append(sub_param)
    print(np.array(params).shape)
    param_list.append(np.array(params))
print(len(param_list))

param_dict = {}
param_dict["h1.weight"] = torch.from_numpy(np.expand_dims(param_list[0],2))
param_dict["h1.bias"] = torch.from_numpy(param_list[1].squeeze())
param_dict["bn1.running_mean"] = torch.from_numpy(param_list[5].squeeze())
param_dict["bn1.running_var"] = torch.from_numpy(param_list[6].squeeze())
param_dict["h2.weight"] = torch.from_numpy(np.expand_dims(param_list[7],2))
param_dict["h2.bias"] = torch.from_numpy(param_list[8].squeeze())
param_dict["bn2.running_mean"] = torch.from_numpy(param_list[12].squeeze())
param_dict["bn2.running_var"] = torch.from_numpy(param_list[13].squeeze())
param_dict["h3.weight"] = torch.from_numpy(np.expand_dims(param_list[14],2))
param_dict["h3.bias"] = torch.from_numpy(param_list[15].squeeze())
param_dict["bn3.running_mean"] = torch.from_numpy(param_list[19].squeeze())
param_dict["bn3.running_var"] = torch.from_numpy(param_list[20].squeeze())
param_dict["h4.weight"] = torch.from_numpy(np.expand_dims(param_list[21],2))
param_dict["h4.bias"] = torch.from_numpy(param_list[22].squeeze())
param_dict["bn4.running_mean"] = torch.from_numpy(param_list[26].squeeze())
param_dict["bn4.running_var"] = torch.from_numpy(param_list[27].squeeze())
param_dict["h5.weight"] = torch.from_numpy(np.expand_dims(param_list[28],2))
param_dict["h5.bias"] = torch.from_numpy(param_list[29].squeeze())
param_dict["bn5.running_mean"] = torch.from_numpy(param_list[33].squeeze())
param_dict["bn5.running_var"] = torch.from_numpy(param_list[34].squeeze())
param_dict["h6.weight"] = torch.from_numpy(param_list[35])
param_dict["h6.bias"] = torch.from_numpy(param_list[36].squeeze())

output = open("xvector_param.pkl","wb")
pickle.dump(param_dict,output)