import pickle as pkl

f_path = "../CSL/csl2020ct_v1.pkl"
fin = pkl.load(open(f_path, "rb"))

split = open("../CSL/split_1.txt", "r")
train_list = []
dev_list = []
test_list = []
for line in split.readlines():
    if "name" in line:
        continue 
    else:
        if "train" in line:
            train_list.append(line.split("|")[0])
        elif "test" in line:
            test_list.append(line.split("|")[0])
        else:
            dev_list.append(line.split("|")[0])
print(len(train_list), len(dev_list), len(test_list))
        
ordered = ['train', 'dev', 'test']
for i, _list in enumerate([train_list, dev_list, test_list]):
    
    gloss_file = open("../CSL/data/%s.gloss"%ordered[i], "w")
    text_file = open("../CSL/data/%s.text"%ordered[i], "w")
    name_file = open("../CSL/data/%s.name"%ordered[i], "w")

    for target in (fin['info']):
        if target['name'] in _list:
        # print(list(target))
            gloss_file.write("\t".join(target['label_gloss']) + "\n")
            text_file.write("".join(target['label_word']) + "\n")
            name_file.write(target['name'] + "\n")
 