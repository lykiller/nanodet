
import json
import os

path = 'E:\\dataset\\soda\\annotations\\'
entries = os.listdir(path)
entries.sort()

main = open(path + entries[0])
main = json.load(main)

for entry in entries[1:]:
    file = open(path + entry)
    file = json.load(file)

    for i in file['images']:
        main['images'].append(i)

    for i in file['annotations']:
        main['annotations'].append(i)
for i in range(len(main['images'])):
    main['images'][i]['id'] = i+1
for i in range(len(main['annotations'])):
    main['annotations'][i]['id'] = i+1

sameID = []
for i in range(0, len(main['annotations'])):
    if(main['annotations'][i]['image_id'] != main['annotations'][i-1]['image_id']):
        sameID.append(main['annotations'][i]['id'])
    #if(main['annotations'][i]['image_id'] == main['annotations'][i-1]['image_id']):

newList = []
c = 1
for i in range(len(sameID)-1):
    newList.extend([c] * (sameID[i+1] - sameID[i]))
    c = c + 1
# newList.extend([75,75,75,75])
for i in range(len(newList)):
    main['annotations'][i]['image_id'] = newList[i]
with open(os.path.join(path, 'merge.json'), 'w') as outfile:
    json.dump(main, outfile)
