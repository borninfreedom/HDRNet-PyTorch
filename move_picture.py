import os
import shutil

# 设置源文件夹路径
source_folder = 'A'

# 获取所有.NEF和.JPG文件
nef_files = [f for f in os.listdir(source_folder) if f.endswith('.NEF')]
jpg_files = [f for f in os.listdir(source_folder) if f.endswith('.JPG')]

# 确保每个子文件夹有200个文件，计算需要创建多少个文件夹
num_nef_files = len(nef_files)
num_jpg_files = len(jpg_files)
num_files_per_folder = 200
num_folders_nef = (num_nef_files + num_files_per_folder - 1) // num_files_per_folder  # 向上取整
num_folders_jpg = (num_jpg_files + num_files_per_folder - 1) // num_files_per_folder  # 向上取整

# 创建子文件夹并移动文件
for i in range(num_folders_nef):
    # 创建子文件夹
    new_folder_nef = os.path.join(source_folder, f'a{i + 1}')
    os.makedirs(new_folder_nef, exist_ok=True)

    # 计算文件索引范围
    start_index = i * num_files_per_folder
    end_index = min(start_index + num_files_per_folder, num_nef_files)

    # 移动.NEF文件
    for file_name in nef_files[start_index:end_index]:
        shutil.move(os.path.join(source_folder, file_name), new_folder_nef)

for i in range(num_folders_jpg):
    # 创建子文件夹
    new_folder_jpg = os.path.join(source_folder, f'b{i + 1}')
    os.makedirs(new_folder_jpg, exist_ok=True)

    # 计算文件索引范围
    start_index = i * num_files_per_folder
    end_index = min(start_index + num_files_per_folder, num_jpg_files)

    # 移动.JPG文件
    for file_name in jpg_files[start_index:end_index]:
        shutil.move(os.path.join(source_folder, file_name), new_folder_jpg)