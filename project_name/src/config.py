import os

current_path = os.path.dirname(os.path.abspath(__file__))
input_files_path = os.path.join(current_path, '..','input')

train_file_path = os.path.join(input_files_path, 'train.csv')
train_fold_file_path = os.path.join(input_files_path, 'train_folds.csv')

n_folds = 5
