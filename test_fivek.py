# #
# # import os
# #
# # # 定义文件夹路径
# # inputs_folder = 'fivek/train/inputs'
# # targets_folder = 'fivek/train/targets'
# #
# # # 获取inputs文件夹中的所有文件名（不包含扩展名）
# # inputs_files = {os.path.splitext(file)[0] for file in os.listdir(inputs_folder) if file.endswith('.dng')}
# #
# # # 获取targets文件夹中的所有文件名（不包含扩展名）
# # targets_files = {os.path.splitext(file)[0] for file in os.listdir(targets_folder) if file.endswith('.tif')}
# #
# # # 检查是否有匹配的同名文件
# # common_files = inputs_files.intersection(targets_files)
# #
# # # 找出在inputs文件夹中但不在targets文件夹中的文件
# # unique_to_inputs = inputs_files - targets_files
# #
# # # 找出在targets文件夹中但不在inputs文件夹中的文件
# # unique_to_targets = targets_files - inputs_files
# #
# # # 打印结果
# # print("Files in 'inputs' folder without corresponding files in 'targets' folder:")
# # for file in unique_to_inputs:
# #     print(file)
# #
# # print("\nFiles in 'targets' folder without corresponding files in 'inputs' folder:")
# # for file in unique_to_targets:
# #     print(file)
# #
# # # 删除这些文件
# # for file in unique_to_inputs:
# #     # 构造完整的文件路径
# #     file_path = os.path.join(inputs_folder, file + '.dng')
# #     # 检查文件是否存在
# #     if os.path.exists(file_path):
# #         os.remove(file_path)
# #         print(f"Deleted: {file_path}")
#
#
# import os
# import shutil
# import random
#
#
# def move_files_with_average_interval(source_folder, destination_folder, num_files):
#     # 获取A文件夹中的所有文件
#     files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
#
#     # 如果文件数量少于要移动的文件数量，直接返回
#     if len(files) <= num_files:
#         print(f"Not enough files in {source_folder} to move {num_files} files.")
#         return
#
#     # 计算间隔
#     interval = len(files) // num_files
#
#     # 选择文件
#     selected_files = []
#     for i in range(num_files):
#         # 使用整除和模运算来确保即使文件总数不能被num_files整除，我们也能均匀地选择文件
#         # 这里我们选择每个间隔的第一个文件，但稍微调整以确保均匀性
#         index = i * interval + (i if len(files) % num_files > i else len(files) % num_files - (num_files - 1 - i))
#         selected_files.append(files[index])
#
#     # 确保没有重复的文件（理论上不应该有，但以防万一）
#     assert len(selected_files) == len(set(selected_files)), "Duplicate files selected, which should not happen."
#
#     # 创建B文件夹（如果不存在）
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     # 移动文件
#     for file_name in selected_files:
#         src_file = os.path.join(source_folder, file_name)
#         dest_file = os.path.join(destination_folder, file_name)
#         shutil.move(src_file, dest_file)
#         print(f"Moved {src_file} to {dest_file}")
#
#
# # 使用示例
# source_folder = 'fivek/train/inputs'  # 替换为你的A文件夹路径
# destination_folder = 'fivek/eval/inputs'  # 替换为你的B文件夹路径
# num_files_to_move = 10  # 要移动的文件数量
#
# move_files_with_average_interval(source_folder, destination_folder, num_files_to_move)


import os
import shutil


def find_and_move_files(folder_a, folder_b, folder_c):
    # 获取A文件夹中的所有文件名（不包含扩展名）
    file_names_a = {os.path.splitext(f)[0] for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))}

    # 创建C文件夹（如果不存在）
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)

    # 遍历B文件夹中的所有文件
    for f in os.listdir(folder_b):
        # 提取B文件夹中文件的文件名（不包含扩展名）
        file_name_b, _ = os.path.splitext(f)

        # 检查文件名是否在A文件夹的文件名集合中
        if file_name_b in file_names_a:
            # 构建源文件和目标文件的完整路径
            src_file = os.path.join(folder_b, f)
            dest_file = os.path.join(folder_c, f)

            # 移动文件到C文件夹（如果同名文件已存在，则会被覆盖）
            shutil.move(src_file, dest_file)
            print(f"Moved {src_file} to {dest_file}")


# 使用示例
folder_a = 'fivek/eval/inputs'  # 替换为你的A文件夹路径
folder_b = 'fivek/train/targets'  # 替换为你的B文件夹路径
folder_c = 'fivek/eval/targets'  # 替换为你的C文件夹路径

find_and_move_files(folder_a, folder_b, folder_c)