import torch
import os, sys
sys.path.append("../")
import pickle
import logging
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import MyDataset
from torch.optim import Adam
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from models import CodeDKT, DKT, PDKT, HELPDKT, CODA, ProblemEncoder, codeProblemRelClassifier, GraphCluster
from utils.utils import RMSE, F1_score, auc, acc
from torch.cuda.amp import GradScaler, autocast
import warnings
warnings.filterwarnings("ignore")

model_map = {
    "CodeDKT": CodeDKT,
    "DKT": DKT,
    "PDKT": PDKT,
    "HELPDKT": HELPDKT,
}

model_args_map = {
    "CodeDKT": {"input_dim": 768, "hidden_dim": 768, "layer_dim":3, "output_dim":1, "NOP":None},
    "DKT": {"input_dim": 768, "hidden_dim": 768, "layer_dim":3, "output_dim":1},
    "PDKT": {"input_dim": 768, "hidden_dim": 768, "layer_dim":3, "output_dim":1, "NOC": None},
    "HELPDKT": {"input_dim": 768, "hidden_dim": 768, "layer_dim":3, "output_dim":1, "NOR": None},
}

class Config():
    def __init__(self, KT_model="DKT", data_name="PDKT", epochs=20, batch_size=16, optimizer="Adam", learning_rate=1e-3, grad_clip=1.0, use_amp=False, denoise=True, graph_cluster=True) -> None:
        self.KT_model = KT_model
        self.data_name = data_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.denoise = denoise
        self.graph_cluster = graph_cluster
        self.model_args = model_args_map[KT_model]
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def __str__(self) -> str:
        return f"KT_model: {self.KT_model}, data_name: {self.data_name}, epochs: {self.epochs}, batch_size: {self.batch_size}, optimizer: {self.optimizer}, learning_rate: {self.learning_rate}, model_args: {self.model_args}"


def compute_metrics(all_pred, all_label):
    rmse = RMSE(all_pred, all_label)
    all_pred = np.round(all_pred)
    rmse2 = RMSE(all_pred, all_label)
    f1 = F1_score(all_pred, all_label)
    auc_v = auc(all_pred, all_label)
    acc_v = acc(all_pred, all_label)
    return rmse, rmse2, f1, auc_v, acc_v

def get_data(config, mode="train", denoise=True, graph_cluster=True, visualize=False):
    """
    mode: train, test, all
    """
    def data_set_cluster(data, mode="train"):
        NOP = len(all_problems)
        problem_encoder = ProblemEncoder(NOP).to(config.device)
        # use ckpt to initialize the model
        problem_encoder.load_state_dict(torch.load(f"../ckpts/PE_{config.data_name}.pth"))
        cprclassifier = codeProblemRelClassifier().to(config.device)
        cprclassifier.load_state_dict(torch.load(f"../ckpts/CPR_{config.data_name}.pth"))
        for student_log in tqdm(data, desc="Graph Clustering..."):
            for problem_log in student_log:
                problem_id = all_problems[problem_log['pro_id']]
                problem_id = torch.tensor(problem_id).to(config.device)
                code_embedding = problem_log['bert_embeddings']
                if mode == 'test' and visualize == False:
                    commit_cluster, A = [GraphCluster(problem_encoder, cprclassifier, problem_id, code_embedding[:i], denoise=denoise) for i in range(1, len(code_embedding)+1)]
                    problem_log["Adjcent_matrix"] = A
                else:
                    commit_cluster = GraphCluster(problem_encoder, cprclassifier, problem_id, code_embedding, denoise=denoise, visualize=visualize)
                problem_log['cluster'] = commit_cluster
        return data
    
    data_path = f"../data/{config.data_name}"
    with open(os.path.join(data_path, "all_concepts.pkl"), "rb") as f:
        all_concepts = pickle.load(f)
    with open(os.path.join(data_path, "all_results.pkl"), "rb") as f:
        all_results = pickle.load(f)
    with open(os.path.join(data_path, "all_problems.pkl"), "rb") as f:
        all_problems = pickle.load(f)

    if mode == "train":
        with open(os.path.join(data_path, f"{mode}.pkl"), 'rb') as f:
            data = pickle.load(f)
        if graph_cluster and not os.path.exists(os.path.join(data_path, f"{mode}_withcluster.pkl")):
            data = data_set_cluster(data)
            with open(os.path.join(data_path, f"{mode}_withcluster.pkl"), 'wb') as f:
                pickle.dump(data, f)            
        else:
            with open(os.path.join(data_path, f"{mode}_withcluster.pkl"), 'rb') as f:
                data = pickle.load(f)
        return data, all_concepts, all_results, all_problems
    elif mode == "test":
        with open(os.path.join(data_path, f"{mode}.pkl"), 'rb') as f:
            data = pickle.load(f)
        # if graph_cluster and not os.path.exists(os.path.join(data_path, f"{mode}_withcluster.pkl")):
        if True:
            data = data_set_cluster(data)
            # with open(os.path.join(data_path, f"{mode}_withcluster.pkl"), 'wb') as f:
            #     pickle.dump(data, f)            
        else:
            with open(os.path.join(data_path, f"{mode}_withcluster.pkl"), 'rb') as f:
                data = pickle.load(f)
        return data, all_concepts, all_results, all_problems
    else:
        with open(os.path.join(data_path, "train.pkl"), "rb") as f:
            train_data = pickle.load(f)
        # if graph_cluster:
        #     train_data = data_set_cluster(train_data)
        if graph_cluster and not os.path.exists(os.path.join(data_path, "train_withcluster.pkl")):
            train_data = data_set_cluster(train_data)
            with open(os.path.join(data_path, "train_withcluster.pkl"), 'wb') as f:
                pickle.dump(train_data, f)            
        else:
            with open(os.path.join(data_path, "train_withcluster.pkl"), 'rb') as f:
                train_data = pickle.load(f)
        with open(os.path.join(data_path, "test.pkl"), "rb") as f:
            test_data = pickle.load(f)
        # if graph_cluster:
        #     test_data = data_set_cluster(test_data, mode='test')
        if graph_cluster and not os.path.exists(os.path.join(data_path, "test_withcluster.pkl")):
            test_data = data_set_cluster(test_data, mode="test")
            with open(os.path.join(data_path, "test_withcluster.pkl"), 'wb') as f:
                pickle.dump(test_data, f)            
        else:
            with open(os.path.join(data_path, "test_withcluster.pkl"), 'rb') as f:
                test_data = pickle.load(f)
        return train_data, test_data, all_concepts, all_results, all_problems
    
# def get_data(mode="train", denoise=False, graph_cluster=False):
#     """
#     mode: train, test, all
#     """

#     def data_set_cluster(data):
#         NOP = len(all_problems)
#         problem_encoder = ProblemEncoder(NOP).to(config.device)
#         cprclassifier = codeProblemRelClassifier().to(config.device)
#         for student_log in data:
#             for problem_log in student_log:
#                 problem_id = problem_log['pro_id']
#                 code_embedding = problem_id['bert_embeddings']
#                 commit_cluster = GraphCluster(problem_encoder, cprclassifier, problem_id, code_embedding, denoise=denoise) 
#                 problem_log['cluster'] = commit_cluster
#         return data

#     def load_data(file_name):
#         with open(os.path.join(data_path, file_name), 'rb') as f:
#             data = pickle.load(f)
#         if graph_cluster:
#             data = data_set_cluster(data)
#         return data

#     data_path = f"../data/{config.data_name}"

#     # Loading common data
#     with open(os.path.join(data_path, "all_concepts.pkl"), "rb") as f:
#         all_concepts = pickle.load(f)
#     with open(os.path.join(data_path, "all_results.pkl"), "rb") as f:
#         all_results = pickle.load(f)
#     with open(os.path.join(data_path, "all_problems.pkl"), "rb") as f:
#         all_problems = pickle.load(f)

#     if mode == "all":
#         train_data = load_data("train.pkl")
#         test_data = load_data("test.pkl")
#         return train_data, test_data, all_concepts, all_results, all_problems
#     else:
#         data = load_data(f"{mode}.pkl")
#         return data, all_concepts, all_results, all_problems



def draw_graph_from_adj_matrix(A):
    G = nx.Graph()
    
    n = len(A)  # 获取邻接矩阵的大小（顶点的数量）
    for i in range(n):
        for j in range(i, n):
            if A[i][j] != 0:  # 如果[i][j]位置的元素不为0，则添加一条边
                G.add_edge(i, j, weight=A[i][j])
    
    pos = nx.spring_layout(G)  # 设置布局
    # 画节点s
    nx.draw_networkx_nodes(G, pos)
    # 画边
    nx.draw_networkx_edges(G, pos)
    # 画标签
    nx.draw_networkx_labels(G, pos)
    # 显示图形
    plt.savefig("graph.png")
    plt.show()

def visualize_tSNE(code_embedding, fusion_embedding):
    tsne = TSNE(n_components=2, random_state=2023)
    # sampled_data = torch.stack(sampled_data).numpy()
    code_data_tsne = tsne.fit_transform(code_embedding.numpy())
    fusion_tsne = tsne.fit_transform(fusion_embedding.numpy())
    plt.scatter(code_data_tsne[:, 0], code_data_tsne[:, 1], s=60, edgecolor='white', linewidths=1)
    plt.axis('off')
    # save the figure to svg file
    plt.savefig('code_tsne.svg', format='svg')
    plt.show()
    plt.scatter(fusion_tsne[:, 0], fusion_tsne[:, 1], s=60, edgecolor='white', linewidths=1)
    plt.axis('off')
    # save the figure to svg file
    plt.savefig('fusion_tsne.svg', format='svg')
    plt.show()

Importance = []
def train(epoch, config, model, optimizer, dataloader):
    def get_loss(output, label):
        # print("output: ", output.size(), "label: ", label.size())
        loss_func = torch.nn.BCEWithLogitsLoss()
        output, label = output[:, :-1].cpu(), label[:, 1:]
        output, label = output.reshape(-1), label.reshape(-1)
        mask = label >= -.9
        label = label[mask].float()
        output = output[mask].float()
        loss = loss_func(output, label)
        return loss

    for w in model.parameters():
        Importance.append(torch.zeros_like(w))
    if epoch == 0:  # web
        All_Importance = []
        Star_vals = []
        for w in model.parameters():
            All_Importance.append(torch.zeros_like(w))
            Star_vals.append(torch.zeros_like(w))
    else:
        model.load_state_dict(
            torch.load(f"../ckpts/CODA_{config.KT_model}_{config.data_name}_visual.pth"))  # 上一次epoch存储的model路径
        All_Importance = torch.load('All_Importance.pth')
        Star_vals = torch.load('Star.pth')
    
    model.train()
    scaler = GradScaler(enabled=config.use_amp)
    loss_list = []
    loop = tqdm(dataloader, desc='Training...')
    batch_num = 0
    for batch in loop:
        batch_num += 1
        optimizer.zero_grad()
        model_input = {}
        if config.graph_cluster:
            cluster = batch[-1]
            batch = batch[:-1]
            cluster = cluster.to(config.device)
        if config.KT_model in ["CodeDKT", "DKT"]:
            input_embedding, label = batch
            input_embedding = input_embedding.to(config.device) 
            model_input["input_embedding"] = input_embedding
        elif config.KT_model in ["PDKT", "HELPDKT"]:
            input_embedding, features, label = batch
            # input_embedding = input_embedding.to(config.device)
            # features = features.to(config.device)
            input_embedding, features = input_embedding.to(config.device), features.to(config.device)
            model_input["input_embedding"] = input_embedding
            model_input["features"] = features

        with autocast(enabled=config.use_amp):
            output = model(cluster, *list(model_input.values())) if config.graph_cluster else model(*list(model_input.values()))
            loss = get_loss(output, label)
        if config.use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

        with torch.no_grad():
            for i, w in enumerate(model.parameters()):
                Importance[i].mul_(batch_num / (batch_num + 1))
                Importance[i].add_(torch.abs(w.grad.data)/(batch_num + 1))

        if epoch > 0:
            for i, w in enumerate(model.parameters()):
                loss += (1 / 2 * torch.sum(torch.mul(All_Importance[i], torch.abs(w - Star_vals[i])))
                         + 1 / 2 * torch.square(
                            torch.sum(torch.mul(All_Importance[i], torch.abs(w - Star_vals[i])))))

        loss_list.append(loss.item())
        # logger.info(f"loss: {sum(loss_list)/len(loss_list)}")
        loop.set_description(f"Training... (loss: {sum(loss_list)/len(loss_list)})")
    return sum(loss_list) / len(loss_list)


def test(config, model, dataloader, all_problems=None):
    model.eval()
    loop = tqdm(dataloader, desc='Testing...')
    all_pred, all_label = [], []
    if config.graph_cluster:
        # need to contruct graph step by step in test phase
        NOP = len(all_problems)
        problem_encoder = ProblemEncoder(NOP).to(config.device)
        # use ckpt to initialize the model
        problem_encoder.load_state_dict(torch.load(f"../ckpts/PE_{config.data_name}.pth"))
        cprclassifier = codeProblemRelClassifier().to(config.device)
        cprclassifier.load_state_dict(torch.load(f"../ckpts/CPR_{config.data_name}.pth"))
    for batch in loop:
        model_input = {}
        if config.graph_cluster:
            cluster = batch[-1]
            batch = batch[:-1]
            cluster = cluster.to(config.device)
        if config.KT_model in ["CodeDKT", "DKT"]:
            input_embedding, label = batch
            input_embedding = input_embedding.to(config.device) 
            model_input["input_embedding"] = input_embedding
        elif config.KT_model in ["PDKT", "HELPDKT"]:
            input_embedding, features, label = batch
            input_embedding, features = input_embedding.to(config.device), features.to(config.device)
            model_input["input_embedding"] = input_embedding
            model_input["features"] = features
        with torch.no_grad():
            output = model(cluster, *list(model_input.values())) if config.graph_cluster else model(*list(model_input.values()))
        output = torch.sigmoid(output)
        # reshape output and label
        output, label = output[:, :-1].cpu(), label[:, 1:]
        output, label = output.reshape(-1), label.reshape(-1)

        all_pred.append(output.numpy())
        all_label.append(label.numpy())
    # compute metrics
    all_pred = np.concatenate(all_pred, axis=0)  # [test_data_nums, max_step]
    all_label = np.concatenate(all_label, axis=0)
    mask = all_label >= -.9
    all_label = all_label[mask]
    all_pred = all_pred[mask]
    logger.info(f"all_pred: {all_pred}, all_label: {all_label}")
    rmse, rmse2, f1, auc_v, acc_v = compute_metrics(all_pred, all_label)
    return {"rmse": rmse, "rmse2": rmse2, "f1": f1, "auc": auc_v, "acc": acc_v}


def visualize(config, choice_idx=0):
    # load pretrain ckpt model from model_visual.pth
    # model forward and set visual argument to True
    # find sequence length larger than 20, compute this sequence adjacent matrix (visualize DAG figure) 
    # and compute sequence commit codebert embedding and fusion embedding to visualize (t-SNE)
    data, all_concepts, all_results, all_problems = get_data(config, mode="test", denoise=config.denoise, graph_cluster=config.graph_cluster, visualize=True)
    if config.KT_model == "CodeDKT":
        extra_params = {"all_problems": all_problems}
    elif config.KT_model == "HELPDKT":
        extra_params = {"all_results": all_results}
    elif config.KT_model == "PDKT":
        extra_params = {"all_concepts": all_concepts}
    else:
        extra_params = {}    
    dataset = MyDataset(data, data_name=config.data_name, KT_model=config.KT_model, mode="test", graph_cluster=config.graph_cluster, **extra_params)
    model_args = config.model_args
    if config.KT_model == "HELPDKT":
        model_args["NOR"] = len(all_results)
    elif config.KT_model == "PDKT":
        model_args["NOC"] = len(all_concepts)
    elif config.KT_model == "CodeDKT":
        model_args["NOP"] = len(all_problems)
    KT_model = model_map[config.KT_model](**model_args)
    KT_model = KT_model.to(config.device)
    # load ckpt to KT model
    model = CODA(KT_model=KT_model).to(config.device)
    model.load_state_dict(torch.load(f"../ckpts/CODA_{config.KT_model}_{config.data_name}_visual.pth"))
    idx = 0
    for code_embedding, results, label, cluster, A in dataset:
        sequence_length = config.max_step - sum(label==-1) 
        if sequence_length > 20:
            # compute adjacent matrix
            draw_graph_from_adj_matrix(A)
            # compute codebert embedding and fusion embedding
            model_input = {}
            model_input['code_embedding'] = code_embedding
            model_input["results"] = results
            code_embedding, fusion_embedding = model(cluster, visual=True, *list(model_input.values()))
            visualize_tSNE(code_embedding, fusion_embedding)
            if idx == choice_idx:
                break 
            else:
                idx += 1


def main(config):
    logger.info(f"config: {config}")
    # build dataloader
    train_data, test_data, all_concepts, all_results, all_problems = get_data(config, mode="all", denoise=config.denoise, graph_cluster=config.graph_cluster)
    if config.KT_model == "CodeDKT":
        extra_params = {"all_problems": all_problems}
    elif config.KT_model == "HELPDKT":
        extra_params = {"all_results": all_results}
    elif config.KT_model == "PDKT":
        extra_params = {"all_concepts": all_concepts}
    else:
        extra_params = {}

    train_dataset = MyDataset(train_data, data_name=config.data_name, KT_model=config.KT_model, mode="train", graph_cluster=config.graph_cluster, **extra_params)
    test_dataset = MyDataset(test_data, data_name=config.data_name, KT_model=config.KT_model, mode="test", graph_cluster=config.graph_cluster, **extra_params)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size*4, shuffle=False)
    # build model
    model_args = config.model_args
    if config.KT_model == "HELPDKT":
        model_args["NOR"] = len(all_results)
    elif config.KT_model == "PDKT":
        model_args["NOC"] = len(all_concepts)
    elif config.KT_model == "CodeDKT":
        model_args["NOP"] = len(all_problems)
    KT_model = model_map[config.KT_model](**model_args)
    KT_model = KT_model.to(config.device)
    if config.graph_cluster:
        # load ckpt to KT model
        KT_model.load_state_dict(torch.load(f"../ckpts/{config.KT_model}_{config.data_name}_best.pth"))
        model = CODA(KT_model=KT_model).to(config.device)
    else:
        model = KT_model
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    if config.graph_cluster:
        # freeze the parameters of KT model 
        for name, param in model.named_parameters():
            if name.startswith("KT_model"):
                param.requires_grad = False

    for epoch in tqdm(range(config.epochs), desc="Training model..."):

        loss = train(epoch, config, model, optimizer, train_dataloader)

        logger.info(f"epoch {epoch}, loss: {loss}")
        metrics = test(config, model, test_dataloader, all_problems=all_problems)
        logger.info(f"epoch {epoch}, metrics: {metrics}")
        # save checkpoint
        best_auc = 0
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            if config.graph_cluster:
                torch.save(model.state_dict(), f"../ckpts/CODA_{config.KT_model}_{config.data_name}_visual.pth")
            else:
                torch.save(model.state_dict(), f"../ckpts/{config.KT_model}_{config.data_name}_best.pth")
    # save final model prediction and label with exercise length of test data
    logger.info("Running Over!")


if __name__ == "__main__":
    config = Config(KT_model="HELPDKT", data_name="PDKT", epochs=15, batch_size=16, optimizer="Adam", learning_rate=1e-3, grad_clip=1.0, use_amp=False, denoise=True, graph_cluster=True)
    # setting logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m%d_%H%M%S")

    task_suffix = config.KT_model + "_" + config.data_name + "_" + formatted_time
    if config.graph_cluster:
        task_suffix = "CODA_" + task_suffix
    file_handler = logging.FileHandler(f'../logs/{task_suffix}.log')
    file_handler.setLevel(logging.INFO)  
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # main(config)
    visualize(config)