import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import CfgNode as CN


def heaviside_activation(x):
    return F.relu(x)


def zero_one_score(logits, targets):
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, targets)


def default_loss_fn(logits, targets):
    return zero_one_score(logits, targets)


class Feedforward(nn.Module):
    """
    A fully-connected neural network that consumes block_size characters to produce the next one.
    """
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'feedforward'
        C.n_embd = None
        C.hidden_dim = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        C.early_stop_delta = 0.01
        C.patience = 5
        C.dropout_prob1 = 0.1
        C.dropout_prob2 = 0.25
        C.choice = "gru"
        return C

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.choice = config.choice
        if self.choice == "gru":
            self.rnn_cell = nn.GRUCell(input_size=config.n_embd,
                                       hidden_size=config.hidden_dim)
        elif self.choice == "rnn":
            self.rnn_cell = nn.RNNCell(input_size=config.n_embd,
                                       hidden_size=config.hidden_dim)
        elif self.choice == "lstm":
            self.rnn_cell = nn.LSTMCell(input_size=config.n_embd,
                                        hidden_size=config.hidden_dim)
        self.logits = nn.Linear(in_features=config.hidden_dim,
                                out_features=config.vocab_size)

        # self.batch_norm1 = nn.BatchNorm1d(config.block_size * config.n_embd)
        # self.dropout1 = nn.Dropout(config.dropout_prob1)
        self.fnn1 = nn.Linear(config.block_size * config.n_embd, config.hidden_dim)
        # self.batch_norm2 = nn.BatchNorm1d(config.hidden_dim)
        # self.dropout2 = nn.Dropout(config.dropout_prob2)
        self.fnn2 = nn.Linear(config.hidden_dim, config.vocab_size)  # allow bias terms
        self.block_size = config.block_size

        # Proper weight initialization
        nn.init.xavier_uniform_(self.rnn_cell.weight_hh)
        nn.init.xavier_uniform_(self.fnn1.weight)
        nn.init.xavier_uniform_(self.fnn2.weight)
        nn.init.xavier_uniform_(self.logits.weight)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.GRUCell, torch.nn.RNN,
                                    torch.nn.GRU, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.Embedding, torch.nn.Dropout,
                                    torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("h") or pn.endswith("_l0"):
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx: torch.LongTensor,
                targets=None,
                loss_fn=default_loss_fn) -> torch.FloatTensor:
        b, max_len = idx.size()
        target_size = (b, max_len)

        pad_rows = target_size[0] - idx.size(0)
        pad_cols = target_size[1] - idx.size(1)

        if pad_cols or pad_rows:
            idx = torch.nn.functional.pad(idx, (0, pad_cols, 0, pad_rows), value=0)

        x_embedded = self.wte(idx)
        hidden_states_list = []
        prev_hidden = torch.zeros(b, 256).to(self.device)

        for t in range(max_len):
            if self.choice in ["gru", "rnn"]:
                hidden_state = self.rnn_cell(x_embedded[:, t, :], prev_hidden)
            elif self.choice == "lstm":
                hidden_state, cell_state = self.rnn_cell(x_embedded[:, t, :])
            hidden_states_list.append(hidden_state)
            prev_hidden = hidden_state

        x = heaviside_activation(hidden_state)
        logits = self.logits(x)

        loss = None
        if targets is not None:
            # The fully connected layer predicts only the last character
            targets = targets[:, -1].reshape(-1)
            loss = loss_fn(logits, targets)
        return logits.reshape(b, 1, -1), loss

    # def forward(self, idx, targets=None, loss_fn=default_loss_fn):
    #     b, t = idx.size()
    #     assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
    #     # print(idx.shape)
    #
    #     tok_emb = self.wte(idx)
    #     # Flatten embeddings in preparation of the fully-connected layers
    #     x = tok_emb.reshape(b, -1)
    #
    #     # 64, 1000 / 1, 650
    #     # 64, 20/ 1, 13
    #     # Target size (p, q)
    #     # x = self.lstm(x)
    #     # x = heaviside_activation(x)
    #
    #     # x = self.batch_norm1(x)
    #     # x = self.dropout1(x)
    #
    #     x = self.fnn1(x)
    #     x = heaviside_activation(x)
    #
    #     # x = self.batch_norm2(x)
    #     # x = self.dropout2(x)
    #
    #     x = self.fnn2(x)
    #     # x = heaviside_activation(x)
    #     # x = self.fnn3(x)
    #
    #     loss = None
    #     if targets is not None:
    #         # The fully connected layer predicts only the last character
    #         targets = targets[:, -1].reshape(-1)
    #         loss = loss_fn(x, targets)
    #
    #     # Reshape output to be consistent with the rest of the training framework
    #     return x.reshape(b, 1, -1), loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False,
                 top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx=idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            # 1, 1, 65
            logits = logits[:, -1, :] / temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=top_k, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
