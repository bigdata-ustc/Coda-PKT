import sys
sys.path.append('../')
sys.setrecursionlimit(100000)

from tree_sitter import Parser
from utils import Constants as C
import re
# from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
import torch
import sys
sys.path.append('../src')
from unixcoder import UniXcoder

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
    file_name = C.code_dir + C.dataset_dir[dataset_name] + str(code_id) + ext_name
    with open(file_name, encoding='utf-8') as f:
        code = f.read()
        code_arr = code.split('\n')
    return code, code_arr

def get_code_tree(dataset_name, code):
    parser = Parser()
    parser.set_language(C.language_selector[dataset_name])
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

# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("microsoft/codebert-base")
# model.to(C.DEVICE)

def get_bert_embeddings(code):
    code_tokens=tokenizer.tokenize(code)
    tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        id = torch.tensor(tokens_ids[:512])[None, :]
        # token_type_ids = torch.zeros(id.shape, dtype=torch.long, device=C.DEVICE)
        context_embedding=model(input_ids=id.to(C.DEVICE))[0].cpu()
        context_embedding = context_embedding.view(-1, 6*C.EMBEDDING_DIM)[0, :]
    return context_embedding

def get_group_embeddings(token_ids):
    with torch.no_grad():
        # token_type_ids = torch.zeros(id.shape, dtype=torch.long, device=C.DEVICE)
        context_embedding=model(input_ids=torch.tensor(token_ids).to(C.DEVICE))[0].cpu()
        context_embedding = context_embedding[:, 0, :].view(-1, 6*C.EMBEDDING_DIM)
    return context_embedding

# unixcoder_model = UniXcoder("microsoft/unixcoder-base-nine")
# unixcoder_model.to(C.DEVICE)

# def get_unixcoder_embeddings(code):
#     token_ids=unixcoder.tokenize(code, max_length=512,mode="<encoder-only>")
#     source_ids=torch.tensor(tokens_ids).to(C.DEVICE)
#     with torch.no_grad():
#         id = torch.tensor(tokens_ids[:512])[None, :]
#         # token_type_ids = torch.zeros(id.shape, dtype=torch.long, device=C.DEVICE)
#         context_embedding=unixcoder_model(input_ids=id.to(C.DEVICE))[0].cpu()
#         context_embedding = context_embedding.view(-1, 6*C.EMBEDDING_DIM)[0, :]
#     return context_embedding

