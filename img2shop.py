import json

json_file = '/home/arthur/Master/SDDM/data/yelp_dataset_photos/photo_id_to_business_id.json'
out_file = 'meta/photo2shop.txt'

with open(json_file) as data_file:
    data = json.load(data_file)

with open(out_file, 'w') as out:
    for entry in data:
        out.write(entry['photo_id'] + " " + entry['business_id'] + " " + entry['label'] + "\n")