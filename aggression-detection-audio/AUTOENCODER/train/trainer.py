import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
from .base_trainer import BaseTrainer

from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, optimizer, resume, config, train_logger)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        self.model.train()
        total_loss = 0
        self.writer.set_step(epoch)
        _trange = tqdm(self.data_loader, leave=True, desc='')
        for batch_idx, batch in enumerate(_trange):
            batch = [b.to(self.device) for b in batch]
            data = batch[:-2]
            data = data if len(data) > 1 else data[0]
            self.optimizer.zero_grad()
            output, target = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                _str = 'Train Epoch: {} Loss: {:.6f}'.format(epoch,loss.item())
                _trange.set_description(_str)
        loss = total_loss / len(self.data_loader)
        self.writer.add_scalar('loss', loss)
        log = {
            'loss': loss,
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0
        self.writer.set_step(epoch, 'valid')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                batch = [b.to(self.device) for b in batch]
                data = batch[:-2]
                data = data if len(data) > 1 else data[0]
                output, target = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
            val_loss = total_val_loss / len(self.valid_data_loader)
            ret = 0
            self.writer.add_scalar('loss', val_loss)
        return {
            'val_loss': val_loss,
            }
