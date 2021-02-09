# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import json
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='python project starter')
    # parser.add_argument('--config', required=True, type=str, help="Please give a config.json file with details")
    # args = parser.parse_args()
    # with open(args.config) as f:
    #     config = json.load(f)
    # params = config['params']
    # exp = config['exp']
    # print("params:{}".format(params))
    # print("exp params:{}".format(exp))

    # print(1001 // 2)
    # print(type(1) is str)
    with open('prob.txt','r') as f:
        lines = f.readlines()
        precisions, recalls = [],[]
        for line in lines:
            pr = line.split(" ")
            precisions.append(float(pr[0]))
            recalls.append(float(pr[1]))
        print(np.mean(precisions),np.std(precisions))
        print(np.mean(recalls), np.std(recalls))