import copy
import json
from src.pre_data import *
from src.expressions_transfer import *

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def expression_from_result(test, num_list, num_stack=None):
    res = []
    for i in test:
        if i[0] == "N":
            res.append(num_list[int(i[1:])])
        else:
            res.append(i)
    return res


value_ac = 0
equation_ac = 0
count = 0

for i in range(5):
    pairs_tested = read_data_json("data/test_"+str(i)+".json")
    result = read_data_json("results/result_"+str(i)+".json")

    tar1 = []
    tar2 = []
    for test_batch in pairs_tested:
        tar1.append([test_batch[0], expression_from_result(test_batch[4], test_batch[6])])
        tar2.append([test_batch[0], expression_from_result(test_batch[5], test_batch[6])])

    for j in range(len(pairs_tested)):
        test_res = result[j]
        test_tar1 = tar1[j]
        test_tar2 = tar2[j]
        if test_res[0] != test_tar1[0]:
            print(j)
        else:
            if test_res[1] == "tree":
                if test_res[2] == test_tar1[1]:
                    equation_ac += 1
                    value_ac += 1
                    count += 1
                else:
                    try:
                        if abs(compute_prefix_expression(test_res[2]) - compute_prefix_expression(test_tar1[1])) < 1e-4:
                            value_ac += 1
                        count += 1
                    except:
                        count += 1
                        continue
            else:
                if test_res[2] == test_tar2[1]:
                    equation_ac += 1
                    value_ac += 1
                    count += 1
                else:
                    try:
                        if abs(compute_postfix_expression(test_res[2]) - compute_postfix_expression(test_tar2[1])) < 1e-4:
                            value_ac += 1
                        count += 1
                    except:
                        count += 1
                        continue

print("test_answer_acc", float(equation_ac)/count, float(value_ac)/count)
