import os
import PIL.Image as Image
import json

root = 'data/dota_1024/test_split/'

test_json = os.path.join(root, 'images')  # test image root
out_file = os.path.join(root, 'test.json')  # test json output path 生成json的位置

data = {}
# 这部分如果不同数据集可以替换
CLASSES = ('plane', 'baseball-diamond', 'bridge',
               'ground-track-field', 'small-vehicle', 'large-vehicle',
               'ship', 'tennis-court', 'basketball-court',
               'storage-tank', 'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter')
data['categories'] = [{"id": 1, "name": "plane", "supercategory": "none"},
                      {"id": 2, "name": "baseball-diamond", "supercategory": "none"},
                      {"id": 3, "name": "bridge", "supercategory": "none"},
                      {"id": 4, "name": "ground-track-field", "supercategory": "none"},
                      {"id": 5, "name": "small-vehicle", "supercategory": "none"},
                      {"id": 6, "name": "large-vehicle", "supercategory": "none"},
                      {"id": 7, "name": "ship", "supercategory": "none"},
                      {"id": 8, "name": "tennis-court", "supercategory": "none"},
                      {"id": 9, "name": "basketball-court", "supercategory": "none"},
                      {"id": 10, "name": "harbor", "supercategory": "none"}, # 数据集的类别
                      {"id": 11, "name": "swimming-pool", "supercategory": "none"},  # 数据集的类别
                      {"id": 12, "name": "helicopter", "supercategory": "none"}]  # 数据集的类别

images = []
for name in os.listdir(test_json):
    file_path = os.path.join(test_json, name)
    file = Image.open(file_path)
    tmp = dict()
    tmp['id'] = name[:-4]

    # idx += 1
    tmp['width'] = file.size[0]
    tmp['height'] = file.size[1]
    tmp['file_name'] = name
    images.append(tmp)

data['images'] = images
with open(out_file, 'w') as f:
    json.dump(data, f)

# with open(out_file, 'r') as f:
#     test = json.load(f)
#     for i in test['categories']:
#         print(i['id'])
print('finish')
