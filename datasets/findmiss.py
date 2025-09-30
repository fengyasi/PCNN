import pandas as pd
import os
# 定义Excel文件和npz文件夹路径
excel_path = './data/EVARlabel.xls'
npz_folder_path = './data/EVARnpz_raw'
# 读取Excel文件的首列
df = pd.read_excel(excel_path, usecols=[0])  # 假设name在首列
names_in_excel = df.iloc[:, 0].astype(str).tolist()
# 获取文件夹中的所有npz文件名（去掉扩展名）
npz_files = [os.path.splitext(file)[0] for file in os.listdir(npz_folder_path)]
# 找出存在于Excel中但不在文件夹中的name
names_not_in_npz_folder = [name for name in npz_files if name not in names_in_excel]
# 打印结果
print("Names in Excel but not in the npz folder:", names_not_in_npz_folder)