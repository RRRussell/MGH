import pandas as pd
import csv

csvFile_1 = open("./DeepBrain_FP_1.csv", "r")
reader_1 = csv.reader(csvFile_1)

result_1 = []
for item in reader_1:
    # 忽略第一行
    if reader_1.line_num == 1:
        continue
    result_1.append([item[0],item[1],item[2],item[3],item[4],item[5]])
csvFile_1.close()
print(result_1)

csvFile_2 = open("./DeepBrain_FP_2.csv", "r")
reader_2 = csv.reader(csvFile_2)

result_2 = []
for item in reader_2:
    # 忽略第一行
    if reader_2.line_num == 1:
        continue
    result_2.append([item[0],item[1],item[2],item[3],item[4],item[5]])
csvFile_2.close()
print(result_2)

csvFile_3 = open("./DeepBrain_FP_3.csv", "r")
reader_3 = csv.reader(csvFile_3)

result_3 = []
for item in reader_3:
    # 忽略第一行
    if reader_3.line_num == 1:
        continue
    result_3.append([item[0],item[1],item[2],item[3],item[4],item[5]])
csvFile_3.close()
print(result_3)

dataframe = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm','probability'])
for i in range(len(result_1)):
	item_1 = result_1[i]
	item_2 = result_2[i]
	item_3 = result_3[i]

	mrn_1 = item_1[0]
	x_1 = item_1[1]
	y_1 = item_1[2]
	z_1 = item_1[3]
	d_1 = item_1[4] 
	p_1 = item_1[5]

	p_2 = item_2[5]
	p_3 = item_3[5]
	if item_1[:4]==item_2[:4]==item_3[:4]:
		new_id = {'seriesuid':mrn_1,'coordX':x_1,'coordY':y_1,'coordZ':z_1,'diameter_mm':d_1,'probability':0.3*float(p_1)+0.4*float(p_2)+0.3*float(p_3)}
		dataframe = dataframe.append(new_id,ignore_index=True)
csv_path = "./"
dataframe.to_csv(csv_path+"DeepBrain_FP_combine.csv",index=False,sep=',',mode='w+')