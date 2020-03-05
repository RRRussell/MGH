import os
import csv
import numpy as np

bbox_path = "./bbox/"
csv_path = "./evaluationScript/dzh_annotations/"

Patients_list = []
detect_d = [0,1000]

for f in os.listdir(bbox_path):
	if "_lbb.npy" in f:
		Patients_list.append(f.replace("_lbb.npy",""))

# write seriesuids.csv
rows = Patients_list
with open (csv_path+"seriesuids.csv","w",encoding="utf-8",newline="") as f:
	writer = csv.writer(f)
	writer.writerows(map(lambda x: [x], rows))

# write annotations_excluded.csv
# 
headers = ["seriesuid","coordX","coordY","coordZ","diameter_mm"]
rows = []
for p in Patients_list:
	label = np.load(bbox_path+p+"_lbb.npy")
	for labelx in label:
		if labelx[3]>detect_d[1] or labelx[3]<detect_d[0]:
			temp = {"seriesuid":p,"coordX":labelx[2],"coordY":labelx[1],"coordZ":labelx[0],"diameter_mm":labelx[3]}
			rows.append(temp)
with open (csv_path+"annotations_excluded.csv","w",encoding="utf-8",newline="") as f:
	writer = csv.DictWriter(f,headers)
	writer.writeheader()
	writer.writerows(rows)

# write annotations.csv
# this csv contains the test data label
headers = ["seriesuid","coordX","coordY","coordZ","diameter_mm"]
rows = []
for p in Patients_list:
	label = np.load(bbox_path+p+"_lbb.npy")
	for labelx in label:
		if labelx[3]<=detect_d[1] and labelx[3]>=detect_d[0]:
			temp = {"seriesuid":p,"coordX":labelx[2],"coordY":labelx[1],"coordZ":labelx[0],"diameter_mm":labelx[3]}
			rows.append(temp)
with open (csv_path+"annotations.csv","w",encoding="utf-8",newline="") as f:
	writer = csv.DictWriter(f,headers)
	writer.writeheader()
	writer.writerows(rows)

# write DeepBrain.csv
# this csv contains the test data prediction
headers = ["seriesuid","coordX","coordY","coordZ","probability"]
rows = []
for p in Patients_list:
	label = np.load(bbox_path+p+"_pbb.npy")
	for labelx in label:
		temp = {"seriesuid":p,"coordX":labelx[3],"coordY":labelx[2],"coordZ":labelx[1],"probability":labelx[0]}
		rows.append(temp)
with open (csv_path+"DeepBrain.csv","w",encoding="utf-8",newline="") as f:
	writer = csv.DictWriter(f,headers)
	writer.writeheader()
	writer.writerows(rows)