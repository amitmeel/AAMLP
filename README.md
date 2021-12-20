# AAMLP
Approaching (Almost) any machine learning problem by Abhishek Thakur

This repo is created to learn from the book  "Approaching (Almost) any machine learning problem by Abhishek Thakur".

To start, create a conda environment using below command:
```
conda create -n environment_name python==3.7
```
once the environment is created, you can activate the environment using below command:
```
conda activate environment_name
```
To deactivate the environment use the below command:
```
conda deactivate
```

and once the environemnt is activated you can install the required python package from pip and conda using below command:
```
conda/pip install package name
```

For this book you can directly create a conda environment using the environments.yml using the below command (for linux users):
```
conda env create -f environments.yml
```
This command will create a virtual environment named aamlp and you can activate it using the below command:
```
conda activate aamlp
```

I'm currently using the windows system and sometimes some python package are not available in conda, so i prefer to install all package with pip using below command (after creating the virtual environment as mentioned at the start):
```
pip install -r requirments.txt
```

for any new ml project, a sample project folder will looks like as shown below:

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

