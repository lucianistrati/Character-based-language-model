import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader


class EarlyStopper:
    def __init__(self, patience=5, delta=0.1):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_metric = float('inf')
        self.early_stop = False
        self.batches_count = 0

    def step(self, metric):
        if metric < self.best_metric - self.delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        return self.counter >= self.patience


class Trainer:
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.num_datapoints = len(train_dataset)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                step_size=200,
                                                       gamma=0.5)
        self.early_stopper = EarlyStopper(patience=5, delta=0.1)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            self.early_stopper.batches_count += 1

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

            if self.early_stopper.batches_count > len(self.train_dataset) / 32:
                self.early_stopper.step(self.loss)
                if self.early_stopper.should_stop():
                    break
