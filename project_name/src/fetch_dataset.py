
## This will fetch the mnist_784 data from the scikit-learn and save it in a csv file

import pandas as pd
import numpy as np
from sklearn import datasets

import config

def fetch_mnist_784():
    pixel_values, target = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    train=pd.concat([pixel_values,target.to_frame(name="label")],axis=1)

    return train


if __name__=="__main__":
    train_data = fetch_mnist_784()
    train_data.to_csv(config.train_file_path,index=False)