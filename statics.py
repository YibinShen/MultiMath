import re
import copy
import json
import numpy as np

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

def compute_prefix_expression_middle(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos1 = re.search("\d+\(", p)
            pos2 = re.search("\)\d+", p)
            if pos1:
                st.append(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:]))
            elif pos2:
                st.append(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()]))
#            pos = re.search("\d+\(", p)
#            if pos:
#                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
            if a / b < 1:
                return "Fraction"
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
            if a - b < 0:
                return "Negative"
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
#            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
#                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

import re
import copy
import json
import numpy as np

from src.expressions_transfer import *

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def expression_from_predix(test, num_list, num_stack=None):
    res = []
    for i in test:
        if i[0] == "N":
            res.append(num_list[int(i[1:])])
        else:
            res.append(i)
    return res

def compute_prefix_expression_middle(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos1 = re.search("\d+\(", p)
            pos2 = re.search("\)\d+", p)
            if pos1:
                st.append(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:]))
            elif pos2:
                st.append(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()]))
#            pos = re.search("\d+\(", p)
#            if pos:
#                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
            if a / b < 1:
                return "Fraction"
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
            if a - b < 0:
                return "Negative"
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
#            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
#                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


num = 0
acc = 0
wrong_list = []
for i in range(5):
#    pairs_trained = read_data_json("data/train_"+str(i)+".json")
    pairs_tested = read_data_json("data/test_"+str(i)+".json")
    result = read_data_json("results/result_"+str(i)+".json")
#    math23k = pairs_tested + pairs_trained
    
    tar1_list = []
    tar2_list = []
    for test_batch in pairs_tested:
        tar1_list.append([test_batch[0], expression_from_result(test_batch[4], test_batch[6])])
        tar2_list.append([test_batch[0], expression_from_result(test_batch[5], test_batch[6])])
    
    test_list = []
    
    for j in range(len(pairs_tested)):
        tar1 = tar1_list[j][:]
        tar2 = tar2_list[j][:]
        res = result[j][:]
        test_batch = pairs_tested[j][:]
        if tar1[0] != test_batch[0]:
            print(j, "   Error!")
        else:
            negative = compute_prefix_expression_middle(tar1[1])
            if negative != None:
                if negative == "Negative" or negative == "Fraction":
                    num += 1
                    if res[1] == "tree":
                        try:
                            if abs(compute_prefix_expression(res[2]) - compute_prefix_expression(tar1[1])) < 1e-4:
                                acc += 1
                            else:
                                wrong_list.append(res)
                        except:
                            continue
                    else:
                        try:
                            if abs(compute_postfix_expression(res[2]) - compute_postfix_expression(tar2[1])) < 1e-4:
                                acc += 1
                                wrong_list.append(res)
                        except:
                            continue
print(acc/num)

