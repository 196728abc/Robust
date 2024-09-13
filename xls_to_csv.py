import os
import pandas as pd

dir='C:/Users/lijia/Desktop/TEST/Point_FLUXCOM_GOSIF/SHR/'
out_dir='C:/Users/lijia/Desktop/TEST/Point_FLUXCOM_GOSIF/SHR/'
for file in os.listdir(dir):
    if os.path.splitext(file)[1] == '.xls':
        newname=os.path.splitext(file)[0]#是为了后边转csv命名
        #print(newname)
        data_xlsx = pd.read_excel(dir+file, index_col=0)
        #print(11)
        data_xlsx.to_csv(out_dir+newname+'.csv',encoding='utf-8')
    else:#如果后缀不是xlsx，那么跳过这个文件
        continue
print('all the excel table have been finished!')