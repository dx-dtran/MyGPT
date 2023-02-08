import json
import os

with open('math.txt', 'w') as write_output:
    for subdir, dirs, files in os.walk('../MATH/train'):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".json"):
                # print(filepath)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    write_output.writelines(data['solution'])

