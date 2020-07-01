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

temp_list = ["* N0 N1", "/ N0 N1", "/ N1 N0", "* N0 - 1 N1", "/ * N0 N1 N2",
             "+ * N0 N1 N2", "* * N0 N1 N2", "/ N1 - 1 N0", "/ - N0 N2 N1", "/ N0 - 1 N1"]
temp_count = [0] * 10
ans_count = [0] * 10

for i in range(5):
    pairs_tested = read_data_json("data/test_"+str(i)+".json")
    result = read_data_json("results/result_"+str(i)+".json")
    
    tar1 = []
    tar2 = []
    temp_tar = []
    for test_batch in pairs_tested:
        tar1.append([test_batch[0], expression_from_result(test_batch[4], test_batch[6])])
        tar2.append([test_batch[0], expression_from_result(test_batch[5], test_batch[6])])
        temp_tar.append([test_batch[0], test_batch[4]])
 
    for j in range(len(pairs_tested)):
        test_res = result[j][:]
        test_tar1 = tar1[j]
        test_tar2 = tar2[j]
        test_temp_tar = temp_tar[j][:]
        
        if " ".join(test_temp_tar[1]) in temp_list:
            temp_index = temp_list.index(" ".join(test_temp_tar[1]))
            temp_count[temp_index] += 1
            
            if test_res[1] == "tree":
                try:
                    if abs(compute_prefix_expression(test_res[2]) - compute_prefix_expression(test_tar1[1])) < 1e-4:
                        ans_count[temp_index] += 1
                except:
                    continue
            if test_res[1] != "tree":
                try:
                    if abs(compute_postfix_expression(test_res[2]) - compute_postfix_expression(test_tar2[1])) < 1e-4:
                        ans_count[temp_index] += 1
                except:
                    continue
                    

print(np.asarray(ans_count) / np.asarray(temp_count) * 100)

op_list = ['+','-','*','/','^']

temp_length_count = [0] * 10
ans_length_count = [0] * 10

for i in range(5):
    pairs_tested = read_data_json("data/test_"+str(i)+".json")
    result = read_data_json("results/result_"+str(i)+".json")
    
    tar1 = []
    tar2 = []
    temp_tar = []
    for test_batch in pairs_tested:
        tar1.append([test_batch[0], expression_from_result(test_batch[4], test_batch[6])])
        tar2.append([test_batch[0], expression_from_result(test_batch[5], test_batch[6])])
        temp_tar.append([test_batch[0], test_batch[4]])
    
    for j in range(len(pairs_tested)):
        test_res = result[j][:]
        test_tar1 = tar1[j]
        test_tar2 = tar2[j]
        test_temp_tar = temp_tar[j][:]
        
        if len(test_tar1[1]) <= 3:
            temp_index = 3
        elif len(test_tar1[1]) >= 9:
            temp_index = 9
        else:
            temp_index =len(test_tar1[1])
        temp_length_count[temp_index] += 1
        
        if test_res[1] == "tree":
            try:
                if abs(compute_prefix_expression(test_res[2]) - compute_prefix_expression(test_tar1[1])) < 1e-4:
                    ans_length_count[temp_index] += 1
            except:
                continue
        if test_res[1] != "tree":
            try:
                if abs(compute_postfix_expression(test_res[2]) - compute_postfix_expression(test_tar2[1])) < 1e-4:
                    ans_length_count[temp_index] += 1
            except:
                continue

print(np.asarray(ans_length_count) / np.asarray(temp_length_count) * 100)

