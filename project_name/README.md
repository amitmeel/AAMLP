We will build a model to classify MNIST dataset.

This project is devided in below structure:

```
├── input
│ ├── train.csv
│ └── test.csv
├── src
│ ├── create_folds.py
│ ├── train.py
│ ├── inference.py
│ ├── models.py
│ ├── config.py
│ └── model_dispatcher.py
├── models
│ ├── model_rf.bin
│ └── model_et.bin
├── notebooks
│ ├── exploration.ipynb
│ └── check_data.ipynb
├── README.md
└── LICENSE
```
where 
* input/: This folder consists of all the input files and data for your machine learning
project.
* src/: We will keep all the python scripts associated with the project here.
* models/: This folder keeps all the trained models.
* notebooks/: All jupyter notebooks (i.e. any *.ipynb file) are stored in the notebooks
folder.
* README.md: This is a markdown file where you can describe your project and
write instructions on how to train the model or to serve this in a production
environment.
* LICENSE: This is a simple text file that consists of a license for the project, such as
MIT, Apache, etc.

To create the dataset, run the train.py file in src folder using below command:
```
python fetch_dataset.py
```
This will create a train.csv file in input folder (make sure input folder do exist) else it witll throw an error.
run the below command:
```
python create_folds.py
```
after applying the StratifiedKFold new dataset will be saved as train_folds.csv. it’s the same as train.csv. The only differences are that this CSV is shuffled and has a new column called kfold. 

to train the model and save it in models directory run the below command:

```
python train.py --fold 0 --model decision_tree_gini

or

python train.py --fold 0 --model decision_tree_entropy
```
This will train the model and dump the trained model in models directory for each fold.

if you want to add any other model, all you need to define that model in model_dispatch.py as key value pair and provide the model name while running train.py in command line.