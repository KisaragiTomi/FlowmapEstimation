import os

mode = 'train'
#mode = 'test'
# 获取指定文件夹中的所有文件名
img = './datasets/nyu/' + mode + '/img'
norm = './datasets/nyu/' + mode + '/norm'
height = './datasets/nyu/' + mode + '/height'
output_file = './data_split/nyu_' + mode + '.txt'
img_names = os.listdir(img)
norm_names = os.listdir(norm)
height_names = os.listdir(height)

count = 0
# 创建并打开输出文件，以写入模式写入文件名+
with open(output_file, 'w') as file:

    for img_name, norm_name, height_name in zip(img_names, norm_names, height_names):
        count += 1
        if count < 2300:
            file.write(mode + '/img/' + img_name + ' ' + mode + '/norm/' +  norm_name + ' ' + mode + '/height/' + height_name + '\n')



# with open("./data_split/nyu_train.txt", 'r') as f:
#     filenames = f.readlines()
#
# dataset_path = './datasets/nyu'
# for sample_path in filenames:
#     img_path = dataset_path + '/' + sample_path.split()[0]
#     norm_path = dataset_path + '/' + sample_path.split()[1]
#     height_path = dataset_path + '/' + sample_path.split()[2]



