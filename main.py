import os
import itertools

from utils import set_seed, setup_logging, CfgNode as CN
from dataset import CharDataset
from trainer import Trainer
from model import Feedforward, zero_one_score


from collections import deque
losses = deque()
early_stopping_counter = 0
best_loss = 9999
all_losses_list = []


import numpy as np
import nltk

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import torch


def calculate_perplexity(predicted_probs, true_token_ids):
    predicted_probs_flat = predicted_probs.view(-1, 65)
    true_labels_flat = true_token_ids.view(-1)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predicted_probs_flat, true_labels_flat)
    perplexity = torch.exp(loss)
    return perplexity


def f1_score(predicted_token_ids, true_token_ids):
    predicted_token_ids = predicted_token_ids.cpu().numpy().tolist()
    true_token_ids = true_token_ids.cpu().numpy().tolist()
    flat_predicted = [item for sublist in predicted_token_ids for item in sublist]
    flat_true = [item for sublist in true_token_ids for item in sublist]
    true_positives = len(set(flat_predicted) & set(flat_true))
    precision = true_positives / len(set(flat_predicted))
    recall = true_positives / len(set(flat_true))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return round(f1, 4)


def bleu_score(predicted_token_ids, true_token_ids):
    predicted_str = [" ".join(map(str, seq)) for seq in predicted_token_ids.tolist()]
    true_str = [" ".join(map(str, seq)) for seq in true_token_ids.tolist()]
    smoothing_function = SmoothingFunction().method1
    bleu = np.mean(
        [sentence_bleu([t], p, smoothing_function=smoothing_function) for t, p in
         zip(true_str, predicted_str)])
    return round(bleu, 4)


def edit_distance(predicted_token_ids, true_token_ids):
    predicted_str = "".join(map(str, predicted_token_ids[0].cpu().numpy()))
    true_str = "".join(map(str, true_token_ids[0].cpu().numpy()))
    distance = np.sum(np.array(list(predicted_str)) != np.array(list(true_str)))
    return round(distance, 4)


def word_error_rate(predicted_token_ids, true_token_ids):
    predicted_str = " ".join(map(str, predicted_token_ids[0].cpu().numpy()))
    true_str = " ".join(map(str, true_token_ids[0].cpu().numpy()))
    distance = nltk.edit_distance(predicted_str, true_str)
    wer = distance / len(true_str.split())
    return round(wer, 4)


def character_error_rate(predicted_token_ids, true_token_ids):
    predicted_str = "".join(map(str, predicted_token_ids[0].cpu().numpy()))
    true_str = "".join(map(str, true_token_ids[0].cpu().numpy()))
    distance = nltk.edit_distance(predicted_str, true_str)
    cer = distance / len(true_str)
    return round(cer, 4)


def pick_best(models, data, loss_fn):
    inputs, targets = data
    scores = []
    for model in models:
        scores.append(model(inputs, targets, loss_fn)[1])
    return models[torch.argmax(torch.tensor(scores))]


def get_val_dataset(data_config, val_data):
    inputs, targets = [], []
    for idx in range(len(val_data) - data_config.block_size):
        inputs.append(val_data[idx:idx+data_config.block_size])
        targets.append(val_data[idx+1:idx+data_config.block_size+1])
    return (torch.tensor(inputs), torch.tensor(targets))


# For debugging purposes if you use IPDB, resolves a multiprocessing issue:
# https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
# You can safely ignore this
__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

# -----------------------------------------------------------------------------

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # model
    C.model = Feedforward.get_default_config()

    # trainer
    C.trainer = CN()
    C.trainer.device = 'auto'
    # dataloder parameters
    C.trainer.num_workers = 4  # 4
    # optimizer parameters
    C.trainer.max_iters = 10000 # 1000
    C.trainer.batch_size = 32  # 64
    C.trainer.learning_rate = 1e-3  # 5e-4
    C.trainer.betas = (0.9, 0.95)
    C.trainer.weight_decay = 0.1  # 0.1 # only applied on matmul weights
    C.trainer.grad_norm_clip = 5.0  # 1.0

    return C


# ---------------------------------------------------------------------------------


def train(config, train_dataset, run_idx):
    print(config)
    setup_logging(config, run_idx)
    set_seed(config.system.seed)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = Feedforward(config.model)

    print("*" * 200)
    print("model", model)
    from torch.nn import Module

    def count_parameters(model: Module) -> int:
        """
        Count the total number of trainable parameters in a PyTorch model.

        Args:
        - model (torch.nn.Module): The PyTorch model.

        Returns:
        - int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Examp
    num_params = count_parameters(model)
    print("number of parameters: ", num_params)
    print("*" * 200)
    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)
    from statistics import mean

    # iteration callback
    def batch_end_callback(trainer):
        global early_stopping_counter, best_loss
        if trainer.iter_num % 10 == 0:
            all_losses_list.append(trainer.loss.item())
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; "
                  f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = ("O God, O God! Come upon us and make peace in this world "
                           "because you have this power")
                temperature = 1.0
                do_sample = True
                top_k = 10
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=temperature, do_sample=do_sample,
                                   top_k=top_k)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print("completion", completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, f"model_{run_idx}.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

        trainer.lr_scheduler.step()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()

    return model


if __name__ == '__main__':
    data_config = CN()
    # If you make this bigger, make sure to give sufficient initial context to the generate method.
    data_config.block_size = 50  # 13  # 20  # 10

    # construct the entire dataset
    with open('data/input.txt', 'r') as f:
        data = f.read()

    # split dataset
    ratio = .8
    split_idx = int(len(data) * ratio)
    train_dataset = CharDataset(data_config, data[:split_idx])

    num_datapoints = len(train_dataset)
    print("Number of datapoints: ", num_datapoints)
    # The validation dataset can be evaluated all at once
    val_data = [train_dataset.stoi[x] for x in data[split_idx:]]
    val_dataset = get_val_dataset(data_config, val_data)

    print(len(train_dataset), len(val_dataset))
    # Set hyperparameter search space
    learning_rates = [1e-3]   # [2e-4, 3e-4]
    hidden_dims = [256]   # [200, 300, 400]
    n_embds = [256]  # [48, 96]
    hyperparameters_list = itertools.product(learning_rates, hidden_dims, n_embds)

    hyperparameters_list = [(learning_rates[0], hidden_dims[0], n_embds[0])]

    # Train a model for each combination of hyperparameters
    trained_models = []
    for (run_idx, (learning_rate, hidden_dim, n_embd)) in enumerate(hyperparameters_list):
        config = get_config()
        config.model.learning_rate = learning_rate
        config.model.hidden_dim = hidden_dim
        config.model.n_embd = n_embd
        trained_models.append(train(config, train_dataset, run_idx))

    # Pick best model according to performance of the provided loss_fn on val_dataset
    selected_model = pick_best(trained_models, val_dataset, loss_fn=zero_one_score)
    import matplotlib.pyplot as plt


    def plot_losses(losses):
        """
        Plot a line chart of losses.

        Args:
        - losses (list): List of loss values at each epoch.
        """
        plt.plot(losses, marker='o')
        plt.title(f'Training Loss Over Epochs '
                  f'{round(100 * 64 * 10 * len(losses) / num_datapoints, 4)} % of data '
                  f'used')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    # Example usage:
    # Assuming 'epoch_losses' is a list containing loss values at each epoch
    plot_losses(all_losses_list)

    # Report results
    predicted_tensor = val_dataset[0]
    true_tensor = val_dataset[1]

    f1_score_value = f1_score(predicted_tensor, true_tensor)
    print(f'F1 Score: {f1_score_value} (Higher is better)')

    bleu_score_value = bleu_score(predicted_tensor, true_tensor)
    print(f'BLEU Score: {bleu_score_value} (Higher is better)')

    edit_distance_value = edit_distance(predicted_tensor, true_tensor)
    print(f'Edit Distance: {edit_distance_value} (Lower is better)')

    wer_value = word_error_rate(predicted_tensor, true_tensor)
    print(f'Word Error Rate: {wer_value} (Lower is better)')

    cer_value = character_error_rate(predicted_tensor, true_tensor)
    print(f'Character Error Rate: {cer_value} (Lower is better)')

    final_accuracy = (selected_model(val_dataset[0], val_dataset[1],
                                     zero_one_score)[1]
                      / len(val_dataset[0]))
    print(f"Final accuracy of best model: {final_accuracy.tolist()} (Higher is better)")

    perplexity_value = calculate_perplexity(predicted_tensor,
                                            true_tensor)
    print(f'Perplexity: {perplexity_value} (Lower is better)')
