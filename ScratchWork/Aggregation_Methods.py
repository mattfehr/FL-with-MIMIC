from torchjd.aggregation import TrimmedMean, Random, Krum, Mean, UPGrad
from collections import OrderedDict
import torch

def FedAVG(model_collection: list[OrderedDict]) -> OrderedDict:
    tmp = model_collection[0]
    result = OrderedDict({k:torch.zeros(size=v.shape) for k,v in tmp.items()})

    for model in model_collection:
        for k,v in model.items():
            result[k] += model[k]

    return OrderedDict({k:v/len(model_collection) for k,v in result.items()})


def agg_TrimmedMean(model_collection: list[OrderedDict]) -> OrderedDict:
    tmp = model_collection[0]
    result = OrderedDict({k:torch.zeros(size=v.shape) for k,v in tmp.items()})

    trim  = TrimmedMean(trim_number = 1)
    #trim  = Random()

    for key in tmp.keys():
        tmp_collect = []
        param_shape = tmp[key].shape

        for model in model_collection:
            #print(type(model[key]),model[key].shape)
            #Reshape it to a single dimension and convert it to a list
            tmp_collect.append( torch.reshape(model[key], shape = (-1,)).tolist()  )

        # Reshape it back onece all the parameters have been aggregated
        result[key] = torch.reshape(trim(matrix= torch.Tensor(tmp_collect)),param_shape)
    
    return result

def Krum_Agg(model_collection: list[OrderedDict]) -> OrderedDict:
    tmp = model_collection[0]
    result = OrderedDict({k:torch.zeros(size=v.shape) for k,v in tmp.items()})

    trim  = Krum(n_byzantine=1,n_selected= len(model_collection) - 1)

    for key in tmp.keys():
        tmp_collect = []
        param_shape = tmp[key].shape

        for model in model_collection:
            #print(type(model[key]),model[key].shape)
            #Reshape it to a single dimension and convert it to a list
            tmp_collect.append( torch.reshape(model[key], shape = (-1,)).tolist()  )

        # Reshape it back onece all the parameters have been aggregated
        result[key] = torch.reshape(trim(matrix= torch.Tensor(tmp_collect)),param_shape)
    
    return result

def Robust_Agg_Template(model_collection: list[OrderedDict]) -> OrderedDict:
    tmp = model_collection[0]
    result = OrderedDict({k:torch.zeros(size=v.shape) for k,v in tmp.items()})

    #trim  = TrimmedMean(trim_number = 1)
    #trim  = Random()
    #trim   = Krum(n_byzantine=1,n_selected= len(model_collection) - 1)
    trim  = Mean()
    #trim = UPGrad()

    tmp = model_collection[0]

    for key in tmp.keys():
        tmp_collect = []
        param_shape = tmp[key].shape

        for model in model_collection:
            #print(type(model[key]),model[key].shape)
            #Reshape it to a single dimension and convert it to a list
            tmp_collect.append( torch.reshape(model[key], shape = (-1,)).tolist()  )

        # Reshape it back onece all the parameters have been aggregated
        result[key] = torch.reshape(trim(matrix= torch.Tensor(tmp_collect)),param_shape)
    
    return result