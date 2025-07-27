import torch
import os, sys
sys.path.append("../")
import pickle
import logging
import datetime
import numpy as np
from data import MyDataset, PretrainDataset
from models import CodeDKT, DKT, PDKT, HELPDKT, ProblemEncoder, codeProblemRelClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
from utils.utils import RMSE, F1_score, auc, acc, contruct_pretrain_classifier_dataset
from torch.cuda.amp import GradScaler, autocast


temperture = 1

class Config():
    def __init__(self, data_name="PDKT", epochs=20, batch_size=16, optimizer="Adam", learning_rate=1e-3, grad_clip=1.0, use_amp=False, method="classifier") -> None:
        self.data_name = data_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.method = method
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    def __str__(self) -> str:
        return f"data_name: {self.data_name}, epochs: {self.epochs}, batch_size: {self.batch_size}, optimizer: {self.optimizer}, learning_rate: {self.learning_rate}"


def compute_metrics(all_pred, all_label):
    rmse = RMSE(all_pred, all_label)
    all_pred = np.round(all_pred)
    rmse2 = RMSE(all_pred, all_label)
    f1 = F1_score(all_pred, all_label)
    auc_v = auc(all_pred, all_label)
    acc_v = acc(all_pred, all_label)
    return rmse, rmse2, f1, auc_v, acc_v

def get_data(mode="train"):
    """
    mode: train, test, all
    """
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
        return data, all_concepts, all_results, all_problems
    elif mode == "test":
        with open(os.path.join(data_path, f"{mode}.pkl"), 'rb') as f:
            data = pickle.load(f)
        return data, all_concepts, all_results, all_problems
    else:
        with open(os.path.join(data_path, "train.pkl"), "rb") as f:
            train_data = pickle.load(f)
        with open(os.path.join(data_path, "test.pkl"), "rb") as f:
            test_data = pickle.load(f)
        return train_data, test_data, all_concepts, all_results, all_problems


def test(config, model, dataloader):
    model.eval()
    loop = tqdm(dataloader, desc='Testing...')
    all_pred, all_label = [], []
    for batch in loop:
        if config.KT_model in ["CodeDKT", "DKT"]:
            input_embedding, label = batch
            input_embedding = input_embedding.to(config.device) 
            with torch.no_grad():
                output = model(input_embedding)
        elif config.KT_model in ["PDKT", "HELPDKT"]:
            input_embedding, features, label = batch
            input_embedding, features = input_embedding.to(config.device), features.to(config.device)
            with torch.no_grad():
                output = model(input_embedding, features)
        output = torch.sigmoid(output)
        # reshape output and label
        output, label = output[:, :-1].cpu(), label[:, 1:]
        output, label = output.reshape(-1), label.reshape(-1)

        all_pred.append(output.numpy())
        all_label.append(label.numpy())
    # compute metrics
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    mask = all_label >= -.9
    all_label = all_label[mask]
    all_pred = all_pred[mask]
    logger.info(f"all_pred: {all_pred}, all_label: {all_label}")
    rmse, rmse2, f1, auc_v, acc_v = compute_metrics(all_pred, all_label)
    return {"rmse": rmse, "rmse2": rmse2, "f1": f1, "auc": auc_v, "acc": acc_v}


def main(config):
    logger.info(f"config: {config}")
    # build dataloader
    train_data, all_concepts, all_results, all_problems = get_data(mode="train")
    NOP = len(all_problems)
    problem_encoder = ProblemEncoder(NOP)
    dataset = contruct_pretrain_classifier_dataset(train_data, all_problems, mode='cls')
    dataset = PretrainDataset(dataset, mode="cls")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # method1: use classifier model compelete classifier task
    model = codeProblemRelClassifier()
    model = model.to(config.device)
    optimizer = Adam(list(problem_encoder.parameters()) + list(model.parameters()), lr=config.learning_rate)
    
    for epoch in tqdm(range(config.epochs), desc="Pretraining classifier model..."):
        loss_list = []
        for batch in dataloader:
            optimizer.zero_grad()
            problem_ids, code_embedding, label = batch
            problem_ids = problem_ids.long()
            # print("problem ids: ", problem_ids)
            problem_embedding = problem_encoder(problem_ids)
            code_embedding, problem_embedding, label = code_embedding.to(config.device), problem_embedding.to(config.device), label.to(config.device)
            output = model(code_embedding, problem_embedding)
            loss_func = torch.nn.BCELoss()
            label = label.float()
            label = label.view(-1, 1)
            loss = loss_func(output, label)
            loss.backward()
            loss_list.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
        logger.info("Epoch: {}, loss: {}".format(epoch, sum(loss_list)/len(loss_list)))

    # # method2: use contrastive learning to pretrain the model
    # dataset = contruct_pretrain_classifier_dataset(train_data, problem_encoder, NOP, mode='CL')
    # dataset = PretrainDataset(dataset, mode='CL')
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # for epoch in tqdm(range(config.epochs), desc="Pretraining contrastive learning model..."):
    #     loss_list = []
    #     for batch in dataloader:
    #         optimizer.zero_grad()
    #         code_embedding, problem_embedding = batch
    #         code_embedding, problem_embedding = code_embedding.to(config.device), problem_embedding.to(config.device)
    #         # output = model(code_embedding, problem_embedding)
    #         scores = torch.einsum("ab,cb->ac",code_embedding, problem_embedding)
    #         loss_func = torch.nn.CrossEntropyLoss()
    #         loss = loss_func(scores/temperture, torch.arange(code_embedding.size(0), device=config.device))
    #         loss.backward()
    #         loss_list.append(loss.item())
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
    #         optimizer.step()
    #     logger.info("Epoch: {}, loss: {}".format(epoch, sum(loss_list)/len(loss_list)))
    torch.save(model.state_dict(), f"../ckpts/CPR_{config.data_name}.pth")
    torch.save(problem_encoder.state_dict(), f"../ckpts/PE_{config.data_name}.pth")
    logger.info("Pretraining Over!")

if __name__ == "__main__":
    config = Config(data_name="AtCoder_C", epochs=20, batch_size=16, optimizer="Adam", learning_rate=1e-3, grad_clip=1.0, use_amp=False)
    # setting logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m%d_%H%M%S")

    task_suffix = "PretrainCLS" + config.data_name + "_" + formatted_time
    file_handler = logging.FileHandler(f'../logs/{task_suffix}.log')
    file_handler.setLevel(logging.INFO)  
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    main(config)