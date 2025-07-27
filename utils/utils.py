import sys
sys.path.append('../')
from sklearn import metrics
sys.setrecursionlimit(100000)

from tree_sitter import Parser, Language
import re
import torch
import pickle 
import random
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('../src')


def RMSE(pred, y):
    return np.sqrt(np.sum((pred-y)*(pred-y))/len(pred))


def F1_score(pred, y):
    f1 = metrics.f1_score(y, pred)
    return f1


def auc(pred_score, true_y, label_value=1):
    f, t, _ = metrics.roc_curve(true_y, pred_score, pos_label=label_value)
    return metrics.auc(f, t)


def acc(pred, y):
    return metrics.accuracy_score(y, pred)

def get_code(source_code):
    code = source_code.split('\n')
    return code


def index_to_token(code, start_point, end_point):
    start_line = start_point[0]
    start_idx = start_point[1]
    end_line = end_point[0]
    end_idx = end_point[1]
    if not code:
        return ''
    if start_line == end_line:
        return code[start_line][start_idx:end_idx]
    else:
        code_token = code[start_line][start_idx:]
        code_token += '\n'
        for i in range(start_line + 1, end_line):
            code_token += code[i]
            code_token += '\n'
        code_token += code[end_line][:end_idx]
        return code_token


def get_node_token(node):
    return '{0}, {1}'.format(node.start_point, node.end_point)


def filter_graph(edge, type):
    return edge[1] == type


def get_parameter(parameter_node):
    if parameter_node.child_by_field_name('declarator'):
        return parameter_node.child_by_field_name('declarator')
    return parameter_node.child_by_field_name('type')


def get_return_statement(root, return_statement):
    return_statement_list = []
    if root.type in return_statement:
        return_statement_list.append(get_node_token(root))
    for child in root.children:
        sub_return_list = get_return_statement(child, return_statement)
        return_statement_list.extend(sub_return_list)
    return return_statement_list


def get_funtion_parameters(root):
    parameters_list = []
    if root.type == 'identifier':
        parameters_list.append(root)
    for child in root.children:
        sub_parameters_list = get_funtion_parameters(child)
        parameters_list.extend(sub_parameters_list)
    return parameters_list


def get_type_graph(edge, token2type, token_arr):
    source = token2type[token_arr[edge[0]]]['name']
    target = token2type[token_arr[edge[2]]]['name']
    new_edge = [source, edge[1], target]
    return new_edge


def get_node_kind(root):
    is_leaf = not len(root.children)
    if is_leaf:
        return type.type
    else:
        inline_type = type.child_by_field_name('inline_type')
        return inline_type.type

def get_parameters_java(root, code):
    parameters = root.child_by_field_name('parameters')
    parameter_arr = []
    if parameters:
        children = root.children
        for child in children:
            if child.type == 'formal_parameter':
                node = {}
                type = root.child_by_field_name('type')
                if not len(type.children):
                    node['type'] = index_to_token(code, type.start_point, type.end_point)
                else:
                    node['type'] = index_to_token(code, type.child_by_field_name('element').start_point, type.child_by_field_name('element').end_point)
                name = root.child_by_field_name('name')
                node['name'] = index_to_token(code, name.start_point, name.end_point)
                parameter_arr.append(node)
    return parameter_arr


def get_sequence(root, sequence, code, flag):
    is_leaf = not len(root.children)
    token = root.type if not is_leaf else index_to_token(code, root.start_point, root.end_point)
    # token = root.type
    if token == 'main':
        flag = True
    for child in root.children:
        get_sequence(child, sequence, code, flag)
    sequence.append(token)

def get_code(code_id, ext_name, dataset_name):
    code_dir = '../data/'
    dataset_dir = {
        'atcoder_c': 'atcoder_c/submissions/',
    }
    file_name = code_dir + dataset_dir[dataset_name] + str(code_id) + ext_name
    with open(file_name, encoding='utf-8') as f:
        code = f.read()
        code_arr = code.split('\n')
    return code, code_arr

def get_code_tree(dataset_name, code):
    parser = Parser()
    C_LANGUAGE = Language('../build/my-languages.so', 'c')
    CPP_LANGUAGE = Language('../build/my-languages.so', 'cpp')
    language_selector = {
        'atcoder_c': C_LANGUAGE,
        'PDKT': C_LANGUAGE,
        'aizu_cpp': CPP_LANGUAGE,
    }
    parser.set_language(language_selector[dataset_name])
    tree = parser.parse(bytes(code, encoding='utf8'))
    root_node = tree.root_node
    return root_node

def get_problem_text(dataset, problem_id):
    url = dataset + '/problems/' + problem_id + '.txt'
    with open(url, encoding='utf-8') as f:
        txt = f.read()
        txt = re.sub('\n', '', txt)
        txt_arr = txt.split(' ')
    return txt, txt_arr

def get_dict(root, code, dict, num):
    token = get_node_token(root)
    if root.type == 'identifier':
        name = index_to_token(code, root.start_point, root.end_point)
        if name in dict.keys():
            dict[name] += 1
        else:
            dict[name] = 1
        num += 1
    if len(root.children):
        for child in root.children:
            dict, num = get_dict(child, code, dict, num)
    return dict, num

def problem_seq(problem, vocab, max_token):
    seq = []
    for word in problem:
        if word in vocab:
            seq.append(vocab[word].index)
        else:
            seq.append(max_token)
    return seq

def get_sequence(code, root, flag):
    sequence = []
    is_leaf = not(len(root.children))
    token = root.type if not is_leaf else index_to_token(code, root.start_point, root.end_point)
    if token == 'main':
        flag = True
    if not is_leaf:
        for child in root.children:
            sub_sequence, sub_flag = get_sequence(code, child, flag)
            sequence.extend(sub_sequence)
            flag = sub_flag
    sequence.append(token)
    return sequence, flag

def get_corpus(code, root):
    sequence = []
    is_leaf = not(len(root.children))
    token = root.type if not is_leaf else index_to_token(code, root.start_point, root.end_point)
    if not is_leaf:
        for child in root.children:
            sub_sequence = get_corpus(code, child)
            sequence.extend(sub_sequence)
    sequence.append(token)
    return sequence

def count_dataset(dataset):
    student_set, problem_set = set(), set()
    submit_num = 0
    avg_submit_num = 0
    avg_solve_num = 0
    student_problem_num = 0
    for student_log in tqdm(dataset, desc=f"Count data..."): # for each student
        for problem_log in student_log:
            if problem_log["sub_num"] <= 2:
                continue
            submit_num += problem_log["sub_num"]
            problem_id = problem_log['pro_id']
            user_id = problem_log['user_id']
            student_problem_num += 1
            avg_solve_num += 1 if sum(problem_log["is_ac_arr"]) > 0 else 0
            if user_id in student_set:
                continue
            else:
                student_set.add(user_id)
            if problem_id in problem_set:
                continue
            else:
                problem_set.add(problem_id)
    avg_solve_num /= student_problem_num
    avg_submit_num = submit_num / student_problem_num
    return len(student_set), len(problem_set), submit_num, avg_solve_num, student_problem_num

def contruct_pretrain_classifier_dataset(data, all_problems, num_neg_samples=16-1, mode="cls"):
    """mode: cls(classification) | CL(contrastive learning)"""
    assert mode in ["cls", "CL"], "mode must be one of cls(classification), CL(contrastive learning)"
    problem2code = {}

    for student_log in tqdm(data, desc=f"Contruct dataset..."): # for each student
        for problem_log in student_log:
            code_embedding = problem_log["bert_embeddings"]
            problem_id = all_problems[problem_log["pro_id"]]
            if problem_id not in problem2code:
                problem2code[problem_id] = []
            # problem_embedding = problem_encoder(problem_log["pro_id"])   # TODO: if need to map origin pro_id to recontruct problem ids
            for idx, submit_status in enumerate(problem_log["is_ac_arr"]):
                if submit_status == True:
                    problem2code[problem_id].append(code_embedding[idx,:])

    all_problems = list(problem2code.keys())
    all_codes = [code for codes in problem2code.values() for code in codes]
    code_to_index = {id(code): idx for idx, code in enumerate(all_codes)}
    samples = []
    for problem, codes in problem2code.items():
        for pos_code in codes:
            # select negative sample
            current_code_indices = [code_to_index[id(code)] for code in codes]

            neg_codes_indices = random.choices([idx for idx in range(len(all_codes)) if idx not in current_code_indices], k=num_neg_samples)
            neg_codes = [all_codes[idx] for idx in neg_codes_indices]
            if mode == "CL":
                samples.append({
                    'problem': problem,
                    'positive_code': pos_code,
                    'negative_codes': neg_codes
                })
            else:
                samples.append({'problem':problem, 'code':pos_code, 'label':1})
                samples += [{'problem': problem, 'code': neg_code, 'label': 0} for neg_code in neg_codes]
        
    return samples


# data = {
#     'query1': ['doc1', 'doc2', 'doc3'],
#     'query2': ['doc4', 'doc5'],
#     'query3': ['doc6', 'doc7', 'doc8', 'doc9'],
#     # ... 可以有更多query和documents
# }

# def construct_samples(data, num_neg_samples=1):
#     all_queries = list(data.keys())
#     all_documents = [doc for docs in data.values() for doc in docs]
    
#     samples = []
    
    # for query, docs in data.items():
    #     # 选择正样本
    #     pos_sample = random.choice(docs)
        
    #     # 选择负样本
    #     neg_samples = random.choices([doc for doc in all_documents if doc not in docs], k=num_neg_samples)
        
    #     samples.append({
    #         'query': query,
    #         'positive_sample': pos_sample,
    #         'negative_samples': neg_samples
    #     })
        
    # return samples

# samples = construct_samples(data, num_neg_samples=2)
# for sample in samples:
#     print(f"Query: {sample['query']}, Positive Sample: {sample['positive_sample']}, Negative Samples: {sample['negative_samples']}")


def get_bert_embeddings(model, tokenizer, code):
    code_tokens=tokenizer.tokenize(code)
    tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        id = torch.tensor(tokens_ids[:512])[None, :]
        context_embedding=model(input_ids=id.to(model.device))[0].cpu()
        context_embedding = context_embedding.view(-1, 768)[0, :]
    return context_embedding


if __name__ == "__main__":
    for data_name in ["PDKT"]:  # , "AtCoder_C"
        data_dir = f"../data/{data_name}/train2.pkl"
        with open(data_dir, "rb") as f:
            train_data = pickle.load(f)
        data_dir = f"../data/{data_name}/test2.pkl"
        with open(data_dir, "rb") as f:
            test_data = pickle.load(f)
        data = train_data + test_data
        student_num, problem_num, submit_num, avg_solve_num, student_problem_num = count_dataset(data)
        print(f"{data_name} -> student num: {student_num}, problem num: {problem_num}, submit num: {submit_num}, avg solve num: {avg_solve_num}, student problem num: {student_problem_num}")