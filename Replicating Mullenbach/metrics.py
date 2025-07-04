from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import torch

@torch.no_grad()
def eval_model(model, device, data_loader,label_space):
    model.eval()
    all_pred   = torch.empty((0,), dtype = torch.float32).to(device)
    all_labels = torch.empty((0,), dtype = torch.float32).to(device)

    for data_inputs, data_labels in data_loader:
        data_inputs = data_inputs.to(device)
        data_labels = data_labels.to(device)
        preds, _ = model(data_inputs)
        pred_labels = F.one_hot(preds.argmax(dim = 1),num_classes=label_space)

        all_pred   = torch.cat((all_pred,pred_labels)  , dim = 0 )
        all_labels = torch.cat((all_labels,data_labels), dim = 0)
    print(classification_report(y_pred =   all_pred.cpu().numpy(), y_true = all_labels.cpu().numpy(), zero_division = 0))

def eval_metrics(y_pred, y_true):
    assert (y_true.shape) == y_pred.shape
    assert(len(y_true.shape)) == 2
    label_count = y_true.shape[1]
    tp_sum      = 0
    tp_fn_sum   = 0
    macro_ratio = 0
    for l_idx in range(label_count):
        l_pred  = y_pred [:,l_idx]
        l_truth = y_true[:,l_idx]
        tn, fp, fn, tp  = confusion_matrix(y_true=l_truth, y_pred=l_pred,labels=[1,0]).ravel()
        #
        tp_sum += tp
        tp_fn_sum += tp + fn
        #
        macro_ratio += tp / (tp + fn)
    micro_ratio  = tp_sum/tp_fn_sum
    macro_ratio /= label_count
    return (micro_ratio,macro_ratio)