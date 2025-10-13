from   ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from   ray import tune
from   ray import train
# from ray.train import Checkpoint, get_checkpoint

from train import GenerateModel, federate_model
from torch.utils.data import DataLoader, TensorDataset
from metrics import eval_model
from functools import partial
import torch
import ray
import os


working_dir  = "/home/drew/FL-with-MIMIC/Replicating Mullenbach/AWS" 
# X_test = torch.load(os.path.join(working_dir,"Data","X_test.pt"))
# Y_test = torch.load(os.path.join(working_dir,"Data","Y_test.pt"))

def load_data():
    working_dir  = "/home/drew/FL-with-MIMIC/Replicating Mullenbach/AWS" 
    X_val  = torch.load(os.path.join(working_dir,"Data","X_val.pt"))
    Y_val  = torch.load(os.path.join(working_dir,"Data","Y_val.pt"))
    return DataLoader(TensorDataset(X_val,Y_val),batch_size=32,shuffle=False)

def train_model(config):
    trained_model = federate_model(config,None)
    working_dir  = "/home/drew/FL-with-MIMIC/Replicating Mullenbach/AWS" 
    model = GenerateModel(table_path = os.path.join(working_dir,"Model","processed_full.w2v"),
                      num_of_filters = config['n_filters'],
                      kernel_size    = config['window_size'])
    model.load_state_dict(trained_model)
    ########################################
    saved_metric = eval_model(
               model       = model,
               device      = torch.device("cpu"),
               data_loader = load_data() # val_loader
               )
    ########################################
    tune.report(saved_metric)

if __name__ == '__main__':
    context = ray.init(dashboard_port = 9012,dashboard_host = '192.168.1.79')
    print(context.dashboard_url)
    config = {
            "batch_size" : tune.choice([8, 16, 32]),
            "lr"         : tune.loguniform(0.0001, 0.1),
            "n_filters"  : tune.choice([i for i in range(5,21)]),
            "window_size": tune.choice([3,4,5,6,7]),
            "epochs"     : tune.choice([2,3,4,5,6]),
            "rounds"     : 2_000
    }
    scheduler = ASHAScheduler(
        metric = "f1_macro",
        mode   = "max",
        max_t  = 24400,
        grace_period     = 100,
        reduction_factor = 2
    )
    

    result = tune.run(
        partial(train_model),
        resources_per_trial={"cpu": 5, "gpu": 1},
        config      = config,
        num_samples = 50, # Number of different combinations
        scheduler   = scheduler,
        storage_path="/home/drew/FL-with-MIMIC/Replicating Mullenbach/AWS/S3_Bucket/results",
        resume=False
    )