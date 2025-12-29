from typing import Optional, Dict, List, Type, Any,Tuple
from .base import CrystalDiffusion,CrystalEM,DiffCSPEM
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


class LitCrystalDiffusion(pl.LightningModule):

    def __init__(
        self,
        model: CrystalDiffusion,
        p_dict: Dict,
    ):
        super().__init__()
        self.p_dict = p_dict
        self.model = model
        self.frac_weight = p_dict['Train']['weight'][0]
        self.cell_weight = p_dict['Train']['weight'][1]
        self.no_valid = p_dict['Train']['noValid']
        # self.train_mode = p_dict["Train"]['warm_up']

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        results = self.model(batch_data)
        return results

    def training_step(self, batch, batch_idx):
        print("!!!!!!!!!!!??????????", torch.is_grad_enabled())
        frac_loss, cell_loss = self(batch)
        loss = self.frac_weight * frac_loss + self.cell_weight
        self.train_loss = loss
        self.log("train_loss", loss)
        self.log("train_frac_loss", frac_loss)
        self.log("train_cell_loss", cell_loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(False)
        frac_loss, cell_loss = self(batch)
        loss = self.frac_weight * frac_loss + self.cell_weight * cell_loss
        self.log("val_loss", 0, batch_size=batch['n_atoms'].shape[0])
        self.log("val_frac_loss", frac_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_cell_loss", cell_loss, batch_size=batch['n_atoms'].shape[0])

    def get_optimizer(self):
        opt_dict = self.p_dict["Train"]["Optimizer"]
        decay_interactions = {}
        no_decay_interactions = {}
        embedding = {}
        readout = {}
        others = {}
        for name, param in self.model.named_parameters():
            if "equivalent" in name:
                if "weight" in name:
                    decay_interactions[name] = param
                else:
                    no_decay_interactions[name] = param
            else:
                if "embedding" in name:
                    embedding[name] = param
                elif "readout" in name:
                    readout[name] = param
                else:
                    others[name] = param
        log.debug(
            f"\nEquivalent weight: {list(decay_interactions.keys())}"
            f"\nEquivalent bias  : {list(no_decay_interactions.keys())}"
            f"\nEmbedding        : {list(embedding.keys())}"
            f"\nReadout          : {list(readout.keys())}"
            f"\nOthers           : {list(others.keys())}"
        )
        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": list(embedding.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": opt_dict["weightDecay"],
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "readouts",
                    "params": list(readout.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "others",
                    "params": list(others.values()),
                    "weight_decay": 0.0,
                },
            ],
            lr=opt_dict["learningRate"],
            amsgrad=opt_dict["amsGrad"],
        )

        if opt_dict['type'] == "Adam":
            return torch.optim.Adam(**param_options)
        elif opt_dict['type'] == "AdamW":
            return torch.optim.AdamW(**param_options)
        else:
            raise Exception("Unsupported optimizer: {}!".format(opt_dict["type"]))

    def get_lr_scheduler(self, optimizer):
        lr_dict = self.p_dict["Train"]["LrScheduler"]
        if lr_dict['type'] == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=lr_dict['gamma']
            )
        elif lr_dict['type'] == "reduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=lr_dict['lrFactor'],
                patience=lr_dict['patience'],
            )
            if isinstance(self.p_dict["Train"]["evalStepInterval"], int):
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "val_loss",
                    "frequency": self.p_dict["Train"]["evalStepInterval"],
                }
            else:
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": self.p_dict["Train"]["evalEpochInterval"],
                }
            return lr_scheduler_config
        elif lr_dict['type'] == "constant":
            return None
        else:
            raise Exception("Unsupported LrScheduler: {}!".format(lr_dict['type']))

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_lr_scheduler(optimizer)
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.p_dict["Train"]["warmupSteps"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / self.p_dict["Train"]["warmupSteps"],
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.p_dict["Train"]["Optimizer"]["learningRate"]

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


class LitCrystalEM(pl.LightningModule):

    def __init__(
        self,
        model: CrystalEM,
        p_dict: Dict,
    ):
        super().__init__()
        self.p_dict = p_dict
        self.model = model
        self.frac_weight = p_dict['Train']['weight'][0]
        self.cell_weight = p_dict['Train']['weight'][1]
        self.CD_weight = p_dict['Train']['weight'][2]
        self.no_valid = p_dict['Train']['noValid']
        self.train_mode = p_dict["Train"]['warm_up']
        # self.automatic_optimization = False

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) ->Tuple[torch.Tensor]:
        if self.train_mode ==0:
            results = self.model.flow(batch_data)
        if self.train_mode ==1:
            results = self.model.energy_matching(batch_data)
        return results


    def training_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True)
        frac_loss, cell_loss,CD_loss = self(batch)
        loss = self.frac_weight * frac_loss + self.cell_weight * cell_loss + self.CD_weight * CD_loss
        self.log("train_loss", loss, batch_size=batch['n_atoms'].shape[0])
        self.log("train_frac_loss", frac_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("train_cell_loss", cell_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("train_CD_loss", CD_loss, batch_size=batch['n_atoms'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        frac_loss, cell_loss,CD_loss = self(batch)
        loss = self.frac_weight * frac_loss + self.cell_weight * cell_loss + self.CD_weight * CD_loss
        self.log("val_loss", loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_frac_loss", frac_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_cell_loss", cell_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_CD_loss", CD_loss, batch_size=batch['n_atoms'].shape[0])
        # del batch_data,frac_loss,cell_loss,CD_loss,loss
        # torch.cuda.empty_cache()

    def get_optimizer(self):
        opt_dict = self.p_dict["Train"]["Optimizer"]
        decay_interactions = {}
        no_decay_interactions = {}
        embedding = {}
        readout = {}
        others = {}
        for name, param in self.model.named_parameters():
            if "equivalent" in name:
                if "weight" in name:
                    decay_interactions[name] = param
                else:
                    no_decay_interactions[name] = param
            else:
                if "embedding" in name:
                    embedding[name] = param
                elif "readout" in name:
                    readout[name] = param
                else:
                    others[name] = param
        log.debug(
            f"\nEquivalent weight: {list(decay_interactions.keys())}"
            f"\nEquivalent bias  : {list(no_decay_interactions.keys())}"
            f"\nEmbedding        : {list(embedding.keys())}"
            f"\nReadout          : {list(readout.keys())}"
            f"\nOthers           : {list(others.keys())}"
        )
        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": list(embedding.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": opt_dict["weightDecay"],
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "readouts",
                    "params": list(readout.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "others",
                    "params": list(others.values()),
                    "weight_decay": 0.0,
                },
            ],
            lr=opt_dict["learningRate"],
            amsgrad=opt_dict["amsGrad"],
        )

        if opt_dict['type'] == "Adam":
            return torch.optim.Adam(**param_options)
        elif opt_dict['type'] == "AdamW":
            return torch.optim.AdamW(**param_options)
        else:
            raise Exception("Unsupported optimizer: {}!".format(opt_dict["type"]))

    def get_lr_scheduler(self, optimizer):
        lr_dict = self.p_dict["Train"]["LrScheduler"]
        if lr_dict['type'] == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=lr_dict['gamma']
            )
        elif lr_dict['type'] == "reduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=lr_dict['lrFactor'],
                patience=lr_dict['patience'],
            )
            if isinstance(self.p_dict["Train"]["evalStepInterval"], int):
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "val_loss",
                    "frequency": self.p_dict["Train"]["evalStepInterval"],
                }
            else:
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": self.p_dict["Train"]["evalEpochInterval"],
                }
            return lr_scheduler_config
        elif lr_dict['type'] == "constant":
            return None
        else:
            raise Exception("Unsupported LrScheduler: {}!".format(lr_dict['type']))

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_lr_scheduler(optimizer)
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.p_dict["Train"]["warmupSteps"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / self.p_dict["Train"]["warmupSteps"],
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.p_dict["Train"]["Optimizer"]["learningRate"]

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value



class LitDiffCSPEM(pl.LightningModule):

    def __init__(
        self,
        model: DiffCSPEM,
        p_dict: Dict,
    ):
        super().__init__()
        self.p_dict = p_dict
        self.model = model
        self.frac_weight = p_dict['Train']['weight'][0]
        self.cell_weight = p_dict['Train']['weight'][1]
        self.CD_weight = p_dict['Train']['weight'][2]
        self.no_valid = p_dict['Train']['noValid']
        self.train_mode = p_dict["Train"]['warm_up']
        # self.automatic_optimization = False

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) ->Tuple[torch.Tensor]:
        if self.train_mode ==0:
            results = self.model.flow(batch_data)
        if self.train_mode ==1:
            results = self.model.energy_matching(batch_data)
        return results


    def training_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True)
        frac_loss, cell_loss,CD_loss = self(batch)
        loss = self.frac_weight * frac_loss + self.cell_weight * cell_loss + self.CD_weight * CD_loss
        self.log("train_loss", loss, batch_size=batch['n_atoms'].shape[0])
        self.log("train_frac_loss", frac_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("train_cell_loss", cell_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("train_CD_loss", CD_loss, batch_size=batch['n_atoms'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        frac_loss, cell_loss,CD_loss = self(batch)
        loss = self.frac_weight * frac_loss + self.cell_weight * cell_loss + self.CD_weight * CD_loss
        self.log("val_loss", loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_frac_loss", frac_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_cell_loss", cell_loss, batch_size=batch['n_atoms'].shape[0])
        self.log("val_CD_loss", CD_loss, batch_size=batch['n_atoms'].shape[0])
        # del batch_data,frac_loss,cell_loss,CD_loss,loss
        # torch.cuda.empty_cache()

    def get_optimizer(self):
        opt_dict = self.p_dict["Train"]["Optimizer"]
        decay_interactions = {}
        no_decay_interactions = {}
        embedding = {}
        readout = {}
        others = {}
        for name, param in self.model.named_parameters():
            if "equivalent" in name:
                if "weight" in name:
                    decay_interactions[name] = param
                else:
                    no_decay_interactions[name] = param
            else:
                if "embedding" in name:
                    embedding[name] = param
                elif "readout" in name:
                    readout[name] = param
                else:
                    others[name] = param
        log.debug(
            f"\nEquivalent weight: {list(decay_interactions.keys())}"
            f"\nEquivalent bias  : {list(no_decay_interactions.keys())}"
            f"\nEmbedding        : {list(embedding.keys())}"
            f"\nReadout          : {list(readout.keys())}"
            f"\nOthers           : {list(others.keys())}"
        )
        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": list(embedding.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": opt_dict["weightDecay"],
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "readouts",
                    "params": list(readout.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "others",
                    "params": list(others.values()),
                    "weight_decay": 0.0,
                },
            ],
            lr=opt_dict["learningRate"],
            amsgrad=opt_dict["amsGrad"],
        )

        if opt_dict['type'] == "Adam":
            return torch.optim.Adam(**param_options)
        elif opt_dict['type'] == "AdamW":
            return torch.optim.AdamW(**param_options)
        else:
            raise Exception("Unsupported optimizer: {}!".format(opt_dict["type"]))

    def get_lr_scheduler(self, optimizer):
        lr_dict = self.p_dict["Train"]["LrScheduler"]
        if lr_dict['type'] == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=lr_dict['gamma']
            )
        elif lr_dict['type'] == "reduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=lr_dict['lrFactor'],
                patience=lr_dict['patience'],
            )
            if isinstance(self.p_dict["Train"]["evalStepInterval"], int):
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "val_loss",
                    "frequency": self.p_dict["Train"]["evalStepInterval"],
                }
            else:
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": self.p_dict["Train"]["evalEpochInterval"],
                }
            return lr_scheduler_config
        elif lr_dict['type'] == "constant":
            return None
        else:
            raise Exception("Unsupported LrScheduler: {}!".format(lr_dict['type']))

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_lr_scheduler(optimizer)
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.p_dict["Train"]["warmupSteps"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / self.p_dict["Train"]["warmupSteps"],
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.p_dict["Train"]["Optimizer"]["learningRate"]