import glob

dir = "shufflenetv2/70"

for filepath in glob.glob(f'{dir}/*.json'):
    print(filepath)
    
    with open(filepath) as f:
        lines = f.read().splitlines()
        
    with open(filepath, "w") as f:
        for i in range(len(lines)):
            line = lines[i]
        
            if i == 0:
                f.write("[" + line + ",\n")
            elif i == len(lines)-1:
                f.write(line + "]\n")
            else:
                f.write(line + ",\n")
