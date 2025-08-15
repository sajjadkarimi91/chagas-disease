clc
clear
close all

% Define paths
db_folder = ['/opt/scratchspace/physionet/physionet2025/official_phase/data/processed_data/training/with-labels/'];
results_path = ['./models/'];


% Create results directory if it doesn't exist
if ~exist(results_path, 'dir')
    mkdir(results_path);
end

train_model(db_folder,results_path, 1)

