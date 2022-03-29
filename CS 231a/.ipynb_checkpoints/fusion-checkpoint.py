import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    with open(file1, "r") as f1:
        data = json.load(f1)
        arr1 = np.asarray(data)
        
    with open(file2, "r") as f2:
        data = json.load(f2)
        arr2 = np.asarray(data)
    
    print(arr1.shape, arr2.shape)
    
    arr1_preds = np.argmax(arr1, axis=1)
    arr2_preds = np.argmax(arr2, axis=1)
    
    mean_scores = np.mean(np.array([arr1, arr2]), axis=0)
    mean_preds= np.argmax(mean_scores, axis=1)
    
    # load ground truths
    DATASET_PATH = '/vision/group/ntu-rgbd/'
    ann_file_test = DATASET_PATH + '50_few_shot_rgb_support_val_ann.txt'
    df = pd.read_csv(ann_file_test, sep=' ', header=None)
    print(df)
    print(len(df))
    y_truth = df.iloc[:,2].to_numpy()
    
    print("Network 1 acc:", accuracy_score(y_truth, arr1_preds))
    print("Network 2 acc:", accuracy_score(y_truth, arr2_preds))
    print("Late fusion acc:", accuracy_score(y_truth, mean_preds))
    
    
    
    

if __name__ == '__main__':
    main()

