filePath = r"3_DeepLearning_Recall.md"
with open(filePath, 'r',encoding='utf-8') as f:
    content = f.readlines()
i = 0
for line in (content):
    line = line.strip()
    if(len(line) > 0):
        print(i ,". ", line, sep = "")
        i+=1