from evaluation      import all_metrics

import torch.nn.functional as F
import torch

@torch.no_grad()
def eval_model(model, device, data_loader):
    model.eval()
    model.to(device)
    all_pred     = torch.empty((0,), dtype = torch.float32).to(device)
    all_labels   = torch.empty((0,), dtype = torch.float32).to(device)
    all_pred_raw = torch.empty((0,), dtype = torch.float32).to(device)

    for data_inputs, data_labels in data_loader:
        data_inputs = data_inputs.to(device)
        data_labels = data_labels.to(device)
        preds, _    = model(data_inputs)
        pred_labels = (F.sigmoid(preds) >= 0.5).long()

        all_pred     = torch.cat((all_pred,pred_labels), dim = 0 )
        all_labels   = torch.cat((all_labels,data_labels), dim = 0)
        all_pred_raw = torch.cat([all_pred_raw, preds], dim = 0)

    #print(classification_report(y_pred =   all_pred.cpu().numpy(), y_true = all_labels.cpu().numpy()))
    return (all_metrics(yhat =   all_pred.cpu().numpy(), y = all_labels.cpu().numpy(), yhat_raw=all_pred_raw.cpu().numpy()  ))

def compare_metric(dict1: dict, dict2: dict, metric: str) -> str:
    """
    AI GENERATED!
    Compares the given metric between two dictionaries and returns a color-coded percent difference.
    
    Parameters:
    - dict1: First dictionary of metrics (e.g., baseline)
    - dict2: Second dictionary of metrics (e.g., new model)
    - metric: The name of the metric to compare
    
    Returns:
    - A string with color-coded percent difference:
        Green:  >5% improvement
        Red:    >5% drop
        Yellow: ~ no significant change
    """
    if metric not in dict1 or metric not in dict2:
        return f"\033[91mError:\033[0m Metric '{metric}' not found in both dictionaries."

    val1 = dict1[metric]
    val2 = dict2[metric]

    if val1 == 0:
        return f"\033[91mError:\033[0m Cannot compute percent difference from zero baseline."

    percent_diff = ((val2 - val1) / abs(val1)) * 100

    # Determine color
    if percent_diff > 5:
        color = "\033[92m"  # Green
    elif percent_diff < -5:
        color = "\033[91m"  # Red
    else:
        color = "\033[93m"  # Yellow

    return f"{color}{metric}: {percent_diff:.2f}% change ({val1:.4f} → {val2:.4f})\033[0m"
