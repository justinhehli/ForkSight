import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.paths import nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join

# Replace with actual paths and configuration names as per your setup
plans_path = '/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed/Dataset001_Segmentation_v1/nnUNetPlans.json'
dataset_json_path = '/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed/Dataset001_Segmentation_v1/dataset.json'
configuration = '2d'
fold = 0

# Load plans and dataset_json
plans = load_json(plans_path)
dataset_json = load_json(dataset_json_path)

# Initialize the PlansManager and ConfigurationManager
plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration(configuration)

# Ensure nnUNet_preprocessed and nnUNet_results are not None
if nnUNet_preprocessed is None or nnUNet_results is None:
    raise ValueError(
        "nnUNet_preprocessed and nnUNet_results must be set to valid paths.")

# Initialize the trainer
trainer = nnUNetTrainer(plans, configuration, fold, dataset_json)
# Initialize the network
trainer.initialize()

# Function to count parameters


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Get the number of parameters
total_params, trainable_params = count_parameters(trainer.network)
print(f'Total parameters in nnUNet model: {total_params}')
print(f'Trainable parameters in nnUNet model: {trainable_params}')
