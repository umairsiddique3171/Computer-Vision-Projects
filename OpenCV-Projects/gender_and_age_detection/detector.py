# importing libraries
import cv2
import os 
import pandas as pd
from deepface import DeepFace


# initializing data_dict
data = {
    "Name" : [],
    "Age" : [],
    "Gender":[]
}

# iterating through files and using deepface analyze method for age and gender detection
for file in os.listdir("imgs"):
    result = DeepFace.analyze(cv2.imread(f"imgs/{file}"),actions = ("age","gender"))
    data["Name"].append(file.split(".")[0])
    data["Age"].append(result[0]["age"])
    data["Gender"].append(result[0]["dominant_gender"])

# converting data_dict into pandas dataframe
df = pd.DataFrame(data)
print(df)

# saving pandas dataframe as csv file
df.to_csv("detections.csv")

