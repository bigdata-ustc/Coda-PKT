import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.cluster import KMeans
import copy
import torch.nn.functional as F
MAX_CODE_LEN = 100


class CodeDKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, NOP):
        super(CodeDKT, self).__init__()
        self.prediction_layer = nn.Linear(input_dim,1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(2*NOP + input_dim,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.linear_fit = nn.Sequential(nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.output_dim))

    def forward(self, inputs, with_coda=False):  
        rnn_input = inputs
        out, hn = self.rnn(rnn_input)  
        if with_coda:
            return out
        res = self.linear_fit(out)
        return res
    

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear_fit = nn.Sequential(nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.output_dim))

    def forward(self, code_vectors, with_coda=False):  
        rnn_input = code_vectors
        
        out, hn = self.rnn(rnn_input)  
        if with_coda:
            return out
        res = self.linear_fit(out).squeeze(2)  
        return res
    

class PDKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, NOC):
        super(PDKT, self).__init__()
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim*2,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.concepts_encoder = nn.Embedding(NOC, self.hidden_dim)
        self.mlp = nn.Linear(input_dim, input_dim)
        self.linear_fit = nn.Sequential(nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.output_dim))

    def forward(self, x, concepts, alpha=0.6, with_coda=False):  
        code_vectors = x
        concept_vectors = self.concepts_encoder(concepts).unsqueeze(1)
        # concept_vectors = concept_vectors
        concept_vectors = concept_vectors.repeat(1, code_vectors.size(1), 1)

        rnn_input = torch.cat((code_vectors, concept_vectors), dim=2)
        out, hn = self.rnn(rnn_input) 

        outputs = []
        for t in range(1, x.size(1) + 1):
            x_t = x[:, t-1]
            mlp_out = self.mlp(x_t)
            sim = torch.matmul(mlp_out.unsqueeze(1), out[:, :t].transpose(1,2)).squeeze(1)
            # exp decay and weight
            exp_decay = torch.exp(-alpha * torch.arange(t-1, -1, -1).float().to(x.device))
            sim *= exp_decay
            weighted_embedding = (F.softmax(sim, dim=1).unsqueeze(2) * out[:, :t]).sum(dim=1)
            outputs.append(weighted_embedding)
        
        output = torch.stack(outputs, dim=1)
        if with_coda:
            return output
        res = self.linear_fit(output)  
        return res
    

class HELPDKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, NOR):
        super(HELPDKT, self).__init__()
        self.attention_softmax = nn.Softmax(dim=1)
        self.NOR = NOR
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.feature_fusion = nn.Linear(3*hidden_dim, hidden_dim)
        self.results_encoder = nn.Embedding(NOR, self.input_dim)
        self.linear_fit = nn.Sequential(nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.output_dim))

    def forward(self, code_vectors, results, with_coda=False):
        # print("code vec: ", code_vectors.shape, "results: ", results.shape)
        result_embedding = self.results_encoder(results)
        # print(f"code vector size: {code_vectors.size()}, result embedding size: {result_embedding.size()}")
        rnn_input = torch.cat((code_vectors, result_embedding), dim=2)
        rnn_input = self.feature_fusion(rnn_input)
        # print("rnn input size: ", rnn_input.size())
        out, hn = self.rnn(rnn_input)
        if with_coda:
            return out
        res = self.linear_fit(out)  
        res = res.view(code_vectors.size(0), -1)
        return res


class ProblemEncoder(nn.Module):
    def __init__(self, NOP, hidden_dim=768):
        super(ProblemEncoder, self).__init__()
        self.embedding_layer = nn.Embedding(NOP, hidden_dim)

    def forward(self, problem_id):
        return self.embedding_layer(problem_id)

# step1: contruct the dataset (positive sample and negative sample), step2: train the classifier
# step1.1: get code and problem embedding (find ac submission in current problem as positive pair, find ac submission in other problem as negative pair)), 
class codeProblemRelCL(nn.Module):
    def __init__(self):
        super(codeProblemRelCL, self).__init__()
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        pass


class codeProblemRelClassifier(nn.Module):
    # method1: use classifier to classify if the code and problem embedding is related
    def __init__(self, hidden_dim=768):
        super(codeProblemRelClassifier, self).__init__()
        self.linear = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, code_embedding, problem_embedding):
        # print("code embedding shape: ", code_embedding.shape, "problem embedding shape: ", problem_embedding.shape)
        pair_embedding = torch.cat((code_embedding, problem_embedding), dim=-1)
        assert pair_embedding.size() == torch.Size([1, 768*2]), f"pair embedding size: {pair_embedding.size()}"
        assert code_embedding.size() == torch.Size([1, 768]), f"code embedding size: {code_embedding.size()}"
        assert problem_embedding.size() == torch.Size([1, 768]), f"problem embedding size: {problem_embedding.size()}"
        res = self.linear(pair_embedding)
        res = self.sigmoid(res)
        return res
    
def reorder_labels(labels):
    
    new_labels = np.zeros_like(labels)
    label_map = {}
    new_label = 0

    for original_label in labels:
        if original_label not in label_map:
            label_map[original_label] = new_label
            new_label += 1
        new_labels[np.where(labels == original_label)] = label_map[original_label]

    return new_labels


def softmax_sim(sim_matrix):
    
    # Convert to Tensor
    sim_matrix = torch.tensor(sim_matrix) 
    
    n = sim_matrix.shape[0]
    
    # Create identity matrix
    identity = torch.eye(n) 
    
    # Set diagonal to -inf 
    sim_matrix = sim_matrix - identity * float("inf")
    
    # Take exponential 
    exp_sim = torch.exp(sim_matrix)
    
    # Take row-wise sum
    row_sums = exp_sim.sum(dim=1).view(-1, 1)
    
    # Divide to get softmax
    softmax_sim = exp_sim / row_sums
    
    return softmax_sim


def softmax_sim(similarity_matrix):
    # Ensure it's a square matrix
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix should be square"

    # Create a mask for the diagonal
    diag_mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)

    # Set diagonal values to negative infinity
    similarity_matrix = similarity_matrix.masked_fill(diag_mask.bool(), float('-inf'))

    # Apply softmax
    softmax_matrix = F.softmax(similarity_matrix, dim=1)

    # If you want to restore the original diagonal values:
    # softmax_matrix = softmax_matrix.masked_scatter(diag_mask.bool(), similarity_matrix.diag())

    return softmax_matrix


def GraphCluster(problem_encoder, CPRClassifier, problem_id, code_embedding, denoise=True, cluster_num=3, alpha=0.8, visualize=False):
    import numpy as np
    n = code_embedding.shape[0]
    if n == 1:
        return [1] if denoise else [0]
    elif n == 2:
        return [1, 2] if denoise else [0, 1]
    code_embedding = torch.tensor(code_embedding).to(problem_id.device)
    code_embedding = torch.nn.functional.normalize(code_embedding, p=2, dim=1)
    # print("embeddings norm shape: ", code_embedding.shape)
    Attn = torch.mm(code_embedding, code_embedding.transpose(0, 1))
    cos_similarity = copy.deepcopy(Attn)
    Attn = softmax_sim(Attn)
    # Set diagonal elements to 1
    Attn.diagonal().fill_(1)
    Agg_embedding = Attn @ code_embedding
    Agg_embedding = Attn @ Agg_embedding
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(np.array(Agg_embedding.cpu()))
    item_cluster = kmeans.labels_
    item_cluster = reorder_labels(item_cluster)
    if denoise:
        # code_embedding = torch.tensor(code_embedding).to(problem_id.device)
        # code_embedding = torch.nn.functional.normalize(code_embedding, p=2, dim=1)
        # # print("embeddings norm shape: ", code_embedding.shape)
        # cos_similarity = torch.mm(code_embedding, code_embedding.transpose(0, 1))
        # Remain the top k value according to cos similarity
        k = int(n**2 * alpha)
        values, indices = torch.flatten(cos_similarity).topk(k)
        cos_similarity = torch.zeros_like(cos_similarity)
        cos_similarity.view(-1)[indices] = values
        AdjcentMatrix = torch.where(cos_similarity > 0, torch.ones_like(cos_similarity), torch.zeros_like(cos_similarity))

        # topK = torch.topk(cos_similarity, k=k)
        # threshold = topK[-1]
        # cos_similarity = torch.where(cos_similarity > threshold, cos_similarity, torch.zeros_like(cos_similarity))
        # Find outliers from the graph (adjacency matrix)
        outliers = []
        for i in range(n):
            if (torch.sum(cos_similarity[i, :i]) + torch.sum(cos_similarity[i, i+1:])) == 0:
                outliers.append(i)
        if outliers:
            problem_embedding = problem_encoder(problem_id)
        # Remove the outlier
        remove_index = []
        for i in outliers:
            # input_embedding = torch.cat((problem_embedding, code_embedding[i,:]), dim=-1)
            cls_output = CPRClassifier(code_embedding[i,:].unsqueeze(0), problem_embedding.unsqueeze(0))
            if cls_output < 0.5:
                remove_index.append(i)
        # Update the cluster label
        for i in range(n):
            if i in outliers:
                item_cluster[i] = 0
            else:
                item_cluster[i] += 1
    if visualize:
        return list(item_cluster), AdjcentMatrix
    return list(item_cluster)


class CODA(nn.Module):
    def __init__(self, KT_model, cluster_nums=3, denoise=True):
        super(CODA, self).__init__()
        self.KT_model = KT_model 
        self.cluster_encoder = nn.Embedding(cluster_nums+1, 768)
        self.fusion_encoder = nn.Linear(768*2, 768)
        self.status_predictor = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.denoise = denoise

    def forward(self, cluster_labels, visualize=False, *KT_input):
        KT_output = self.KT_model(*KT_input, with_coda=True)
        # sequence_length = torch.sum(cluster_labels == -1, dim=1)
        cluster_labels = torch.where(cluster_labels == -1, torch.zeros_like(cluster_labels), cluster_labels)
        cluster_embedding = self.cluster_encoder(cluster_labels)
        # if self.denoise:
        #     for cluster_label in cluster_labels:
        #         cluster_embedding[cluster_label] += KT_output
        # print("KT output: ", KT_output.shape, "cluster embedding: ", cluster_embedding.shape)
        fusion_embedding = self.fusion_encoder(torch.cat((KT_output, cluster_embedding), dim=-1))
        status = self.status_predictor(fusion_embedding)
        if visualize:
            return KT_input[0], fusion_embedding
        return status

# step1. deal with dataset with graph cluster and denoise, step2. use cluster information to enchance the prediction of student performance
# def contruct_dataset(data, NOP):
#     from tqdm import tqdm
#     positive_pairs = []
#     problem_encoder = ProblemEncoder(NOP)
#     for student_log in tqdm(data, desc=f"Contruct dataset..."): # for each student
#         for problem_log in student_log:
#             code_embedding = problem_log["bert_embeddings"]
#             problem_embedding = problem_encoder(problem_log["pro_id"])
#             for idx, submit_status in enumerate(problem_log["is_ac_arr"]):
#                 if submit_status == True:
#                     positive_pairs.append((code_embedding[idx], problem_embedding))

#     return positive_pairs                    