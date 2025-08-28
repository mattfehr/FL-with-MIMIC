import sagemaker
from sagemaker.tensorflow import TensorFlow
import boto3
from sagemaker.inputs import TrainingInput
 
# Define your SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()  # Or use your IAM role ARN
s3_input_path = "s3://sensoride-ml/ml-data/v112_MLP_xcorr_on_noise_on_rotation_on_per_row_normalzition_on_cfar_on_bike_car_2D_sparse23.mat"
# Extract filename from S3 path
mat_filename = s3_input_path.split("/")[-1]
 
 
hyperparams = {
    "learning-rate": 1e-4,
    "neurons": 512,
    "dropout-rate": 0.1,
    "num-layers": 10,
    "bottleneck-neurons": 256,
    "epochs": 200,
    "batch-size": 512,
    "mat-file": mat_filename,
    "model-name": "mlp_mid2high_v112_sparse23.h5"
}
 
 
 
# Define the TensorFlow Estimator
estimator = TensorFlow(
    entry_point="mlp_train.py",
    source_dir="src",
    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    framework_version="2.12",
    py_version="py310",
    hyperparameters=hyperparams,
    output_path="s3://sensoride-ml/model-output"
)
 
inputs = {
    "training": TrainingInput(s3_input_path, content_type="application/x-matlab-data")
}
# Launch training
estimator.fit(inputs)