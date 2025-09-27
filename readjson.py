import json

with open('data/matrix_city/aerial/train/block_all/estimated_depth_scales.json', 'r') as f:
    data = json.load(f)
    for i in data:
        data[i]['scale']=1
        data[i]['offset']=0
    with open('data/matrix_city/aerial/train/block_all/estimated_depth_scales1.json', 'w') as k:
        json.dump(data, k,indent=4)
        print("done")