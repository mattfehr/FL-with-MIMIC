from sklearn.metrics import classification_report
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

        all_pred   = torch.cat((all_pred,pred_labels), dim = 0 )
        all_labels = torch.cat((all_labels,data_labels))
    print(classification_report(y_pred =   all_pred.cpu().numpy(), y_true = all_labels.cpu().numpy(), zero_division = 0))