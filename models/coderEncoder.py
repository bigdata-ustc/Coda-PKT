import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class Code2Vec(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class BERTEmbedding():
    def __init__(self, model_name, device) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.embedding_dim = 768

    def get_bert_embeddings(self, code):
        code_tokens=self.tokenizer.tokenize(code)
        tokens=[self.tokenizer.cls_token]+code_tokens+[self.tokenizer.sep_token]
        tokens_ids=self.tokenizer.convert_tokens_to_ids(tokens)
        with torch.no_grad():
            id = torch.tensor(tokens_ids[:512])[None, :]
            # token_type_ids = torch.zeros(id.shape, dtype=torch.long, device=C.DEVICE)
            context_embedding=self.model(input_ids=id.to(self.device))[0].cpu()
            context_embedding = context_embedding.view(-1, self.embedding_dim)[0, :]
        return context_embedding

    def get_group_embeddings(self, token_ids):
        with torch.no_grad():
            # token_type_ids = torch.zeros(id.shape, dtype=torch.long, device=C.DEVICE)
            context_embedding=self.model(input_ids=torch.tensor(token_ids).to(self.device))[0].cpu()
            context_embedding = context_embedding[:, 0, :].view(-1, self.embedding_dim)
        return context_embedding

