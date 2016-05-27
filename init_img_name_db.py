import json

IMG2SHOP_FILE = 'meta/photo_id_to_business_id.json'
OUT_FILE = 'meta/img_names.txt'

with open(IMG2SHOP_FILE) as in_file:
    json_list = json.load(in_file)
    with open(OUT_FILE, 'w') as out_file:
        for item in json_list:
            img_id = item['photo_id']
            out_file.write(img_id + "\n")