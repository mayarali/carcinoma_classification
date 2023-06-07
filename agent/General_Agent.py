from dataloader import OxML_Supervised_Dataloader
from tqdm import tqdm
import torch
from colorama import Fore
import time
import torch.nn as nn
import copy
import torch.optim as optim
from extra_utils.WarmupScheduler import WarmupScheduler
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import numpy as np
from collections import defaultdict
from models.Simple_CNN import Simple_CNN
import csv


class General_Agent():
    def __init__(self, config):
        self.config = config

        self.dataloaders = globals()[self.config.dataset.dataloader_class](config)
        self.init_model_opt()
        self.init_logs()
        self.init_loss()


    def run(self):
        if self.config.model.load_ongoing:
            self.load_model_logs(self.config.model.save_dir)

        self.train()

        self.model.load_state_dict(self.best_model.state_dict())
        print(self.test())
        self.test_unlabelled()

    def init_logs(self):
        self.steps_no_improve = 0

        if "weights" not in vars(self).keys(): self.weights = None

        self.logs = {"current_epoch":0,"current_step":0,"steps_no_improve":0, "saved_step": 0, "train_logs":{},"val_logs":{},"test_logs":{},"best_logs":{"val_loss":{"total":100}} , "seed":self.config.training_params.seed, "weights": self.weights}
        # if self.config.training_params.wandb_disable:
        #     self.wandb_run = wandb.init(reinit=True, project="sleep_transformers", config=self.config, mode = "disabled", dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb")
        # else:
        #     self.wandb_run = wandb.init(reinit=True, project="sleep_transformers", config=self.config, dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb" )

        self.device = "cuda:{}".format(self.config.training_params.gpu_device[0])

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def init_model_opt(self):
        model_class = globals()[self.config.model.model_class]
        self.model = model_class(args = self.config.model.args)
        self.model = nn.DataParallel(model_class(args = self.config.model.args), device_ids=[torch.device(i) for i in self.config.training_params.gpu_device])
        self.best_model = copy.deepcopy(self.model)
        self._my_numel(self.model, verbose=True)

        self.optimizer = optim.Adam(self.model.parameters(),
                                          lr=self.config.optimizer.learning_rate,
                                          betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                                          eps=1e-07,
                                          weight_decay=self.config.optimizer.weight_decay)

        after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=4, T_mult=2)
        self.scheduler = WarmupScheduler(optimizer=self.optimizer,
                                               base_lr=self.config.optimizer.learning_rate,
                                               n_warmup_steps=self.config.scheduler.warm_up_steps,
                                               after_scheduler=after_scheduler)

        # wandb.watch(self.model, log_freq=100)


    def train(self):

        # for batch in self.dataloaders.train_loader:
        #     print(batch["data"].shape)
        #     print(batch["label"].shape)
        self.model.train()
        # self._freeze_encoders(config_model=self.agent.config.model, model=self.agent.model)
        self._my_numel(self.model, only_trainable=True)
        self.start = time.time()

        self.running_values = {
            "targets": [],
            "preds": [],
            "batch_loss": [],
            "cond_speed": [],
            "early_stop": False,
            "saved_at_step": 0,
            "prev_epoch_time": 0,
            "val_loss": {"combined":0}
        }
        for self.logs["current_epoch"] in range(self.logs["current_epoch"], self.config.early_stopping.max_epoch):
            pbar = tqdm(enumerate(self.dataloaders.train_loader), desc="Training", leave=None, disable=self.config.training_params.tdqm_disable, position=0)
            for batch_idx, served_dict in pbar:

                self.optimizer.zero_grad()

                data = served_dict["data"].float().to(self.device)
                label = served_dict["label"].squeeze().type(torch.LongTensor).to(self.device)

                pred = self.model(data)

                total_loss = self.loss(pred, label)
                total_loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1)
                self.optimizer.step()
                self.scheduler.step(self.logs["current_step"]+1)

                self.running_values["batch_loss"].append({"total":total_loss.detach().cpu().numpy()})
                self.running_values["targets"].append(label)
                self.running_values["preds"].append({"total": pred.detach().cpu().numpy()})

                # del served_dict, step_outcome
                torch.cuda.empty_cache()

                pbar_message = self.local_logging(batch_idx)

                pbar.set_description(pbar_message)
                pbar.refresh()
                if self.running_values["early_stop"]: return
                self.logs["current_step"] += 1

    def validate(self):
        self.model.eval()
        self.running_values = {
            "targets": [],
            "preds": [],
            "batch_loss": []
        }
        with torch.no_grad():
            pbar = tqdm(enumerate(self.dataloaders.valid_loader), desc="Test", leave=False,
                        disable=True, position=1)
            for batch_idx, served_dict in pbar:

                data = served_dict["data"].float().to(self.device)
                label = served_dict["label"].squeeze().type(torch.LongTensor).to(self.device)

                pred = self.model(data)

                total_loss = self.loss(pred, label)

                self.running_values["batch_loss"].append({"total": total_loss.detach().cpu().numpy()})
                self.running_values["targets"].append(label)
                self.running_values["preds"].append({"total": pred.detach().cpu().numpy()})

                del served_dict

                pbar_message = "Validation batch {0:d}/{1:d} with ".format(batch_idx,
                                                                           len(self.dataloaders.valid_loader) - 1)

                mean_batch = self._calc_mean_batch_loss(batch_loss=self.running_values["batch_loss"])

                for mean_key in mean_batch: pbar_message += "{}: {:.3f} ".format(mean_key, mean_batch[mean_key])
                pbar.set_description(pbar_message)
                pbar.refresh()

            self.running_values["targets"] = torch.cat(self.running_values["targets"]).cpu().numpy()

            total_preds, val_metrics = {}, defaultdict(dict)
            val_metrics["val_loss"] = dict(mean_batch)
            for pred_key in self.running_values["preds"][0]:
                total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in self.running_values["preds"]],
                                                       axis=0).argmax(axis=-1)
                val_metrics["val_acc"][pred_key] = np.equal(self.running_values["targets"],
                                                             total_preds[pred_key]).sum() / len(
                    self.running_values["targets"])
                val_metrics["val_f1"][pred_key] = f1_score(total_preds[pred_key], self.running_values["targets"],
                                                            average="macro")
                val_metrics["val_k"][pred_key] = cohen_kappa_score(total_preds[pred_key],
                                                                    self.running_values["targets"])
                val_metrics["val_perclassf1"][pred_key] = f1_score(total_preds[pred_key],
                                                                    self.running_values["targets"], average=None)
            val_metrics = dict(val_metrics)  # Avoid passing empty dicts to logs, better return an error!

        return val_metrics

    def test(self):
        self.model.eval()
        self.running_values = {
            "targets": [],
            "preds": [],
            "batch_loss": []
        }
        with torch.no_grad():
            pbar = tqdm(enumerate(self.dataloaders.test_loader), desc="Test", leave=False,
                        disable=True, position=1)
            for batch_idx, served_dict in pbar:

                data = served_dict["data"].float().to(self.device)
                label = served_dict["label"].squeeze().type(torch.LongTensor).to(self.device)

                pred = self.model(data)

                total_loss = self.loss(pred, label)

                self.running_values["batch_loss"].append({"total":total_loss.detach().cpu().numpy()})
                self.running_values["targets"].append(label)
                self.running_values["preds"].append({"total": pred.detach().cpu().numpy()})

                # inits.append(init.flatten())

                del served_dict

                pbar_message = "Test batch {0:d}/{1:d} with ".format(batch_idx, len(self.dataloaders.test_loader) - 1)

                mean_batch = self._calc_mean_batch_loss(batch_loss=self.running_values["batch_loss"])

                for mean_key in mean_batch: pbar_message += "{}: {:.3f} ".format(mean_key, mean_batch[mean_key])
                pbar.set_description(pbar_message)
                pbar.refresh()

            self.running_values["targets"] = torch.cat(self.running_values["targets"]).cpu().numpy()

            total_preds, test_metrics = {}, defaultdict(dict)
            test_metrics["test_loss"] = dict(mean_batch)
            for pred_key in self.running_values["preds"][0]:
                total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in self.running_values["preds"]],
                                                       axis=0).argmax(axis=-1)
                test_metrics["test_acc"][pred_key] = np.equal(self.running_values["targets"],
                                                             total_preds[pred_key]).sum() / len(
                    self.running_values["targets"])
                test_metrics["test_f1"][pred_key] = f1_score(total_preds[pred_key], self.running_values["targets"],
                                                            average="macro")
                test_metrics["test_k"][pred_key] = cohen_kappa_score(total_preds[pred_key],
                                                                    self.running_values["targets"])
                test_metrics["test_perclassf1"][pred_key] = f1_score(total_preds[pred_key],
                                                                    self.running_values["targets"], average=None)
            test_metrics = dict(test_metrics)  # Avoid passing empty dicts to logs, better return an error!
            # print(val_metrics)
        return test_metrics

    def test_unlabelled(self):
        self.best_model.eval()
        results = {}
        with torch.no_grad():
            pbar = tqdm(enumerate(self.dataloaders.test_loader_unlabelled), desc="Test Unlabelled", leave=False,
                        disable=True,
                        position=1)
            for batch_idx, served_dict in pbar:
                data = served_dict["data"].float().to(self.device)
                pred = self.model(data)
                results.update({served_dict["id"][i].item(): pred[i].argmax(dim=0).detach().cpu().item() - 1 for i in
                                range(len(pred))})

        with open(self.config.model.test_unlabelled_savedir, 'w') as csvfile:
            csvfile.write("%s,%s\n" % ("id", "malignant"))
            for key in results.keys():
                csvfile.write("%s,%s\n" % (key, results[key]))
    def load_model_logs(self, file_name):

        print("Loading from file {}".format(file_name))
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.logs = checkpoint["logs"]
        # if hasattr(checkpoint, "metrics"):
        #     self.data_loader.load_metrics_ongoing(checkpoint["metrics"])
        # if hasattr(self.logs, "weights"):
        #     self.data_loader.weights = self.logs["weights"]
        #     self.weights = self.agent.data_loader.weights

        # for step in self.logs["train_logs"]:
        #     wandb.log({"train": self.logs["train_logs"][step], "val": self.logs["val_logs"][step]},
        #               step=step)
        #     for i, lr in enumerate(self.logs["train_logs"][step]["learning_rate"]):
        #         wandb.log({"lr": lr, "val": self.logs["val_logs"][step]},
        #                   step=i + step - self.config.early_stopping.validate_every)

        self.loss = nn.CrossEntropyLoss()

        print("Model has loaded successfully")
        print("Metrics have been loaded")
        print("Loaded loss weights are:", self.weights)

        message = Fore.WHITE + "The best in step: {} so far \n".format(
            int(self.logs["best_logs"]["step"] / self.config.early_stopping.validate_every))

        if "val_loss" in self.logs["best_logs"]:
            for i, v in self.logs["best_logs"][
                "val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
        if "val_acc" in self.logs["best_logs"]:
            for i, v in self.logs["best_logs"][
                "val_acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
        if "val_f1" in self.logs["best_logs"]:
            for i, v in self.logs["best_logs"][
                "val_f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
        if "val_k" in self.logs["best_logs"]:
            for i, v in self.logs["best_logs"][
                "val_k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
        if "val_perclassf1" in self.logs["best_logs"]:
            for i, v in self.logs["best_logs"][
                "val_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(i, "{}".format(
                str(list((v * 100).round(2)))))

        print(message)

    def save_model_logs(self, file_name, verbose=False):

        save_dict = {}
        savior = {}
        savior["model_state_dict"] = self.model.state_dict()
        savior["best_model_state_dict"] = self.best_model.state_dict()
        savior["optimizer_state_dict"] = self.optimizer.state_dict()
        savior["logs"] = self.logs
        if hasattr(self.dataloaders, "metrics"):
            savior["metrics"] = self.dataloaders.metrics
        savior["configs"] = self.config

        save_dict.update(savior)

        if self.config.dataset.data_split.split_method == "patients_folds": file_name = file_name.format(
            self.config.dataset.data_split.fold)

        try:
            torch.save(save_dict, file_name)
            second_filename = file_name.split(".")[-3] + "_cp.pth.tar"
            torch.save(save_dict, second_filename)
            if verbose:
                print(Fore.WHITE + "Models has saved successfully in {}".format(file_name))
        except:
            raise Exception("Problem in model saving")

    def local_logging(self, batch_idx):
        pbar_message = Fore.WHITE + "Training batch {0:d}/{1:d} steps no improve {2:d} with ".format(batch_idx,
                                                                                                     len(self.dataloaders.train_loader) - 1,
                                                                                                     self.logs[
                                                                                                         "steps_no_improve"])
        mean_batch = self._calc_mean_batch_loss(batch_loss=self.running_values["batch_loss"])
        for mean_key in mean_batch: pbar_message += "{}: {:.3f} ".format(mean_key, mean_batch[mean_key])

        if self.logs["current_step"] % self.config.early_stopping.validate_every == 0 and \
                self.logs["current_step"] // self.config.early_stopping.validate_every >= self.config.early_stopping.validate_after and \
                batch_idx != 0:

            early_stop, val_loss = self.checkpointing(batch_loss=mean_batch, predictions=self.running_values["preds"], targets=self.running_values["targets"])
            self.running_values.update({ "targets": [], "preds": [], "batch_loss": [], "early_stop": early_stop, "val_loss": val_loss})
            if self.running_values["early_stop"]: return
            self.running_values["saved_at_step"] = self.logs["saved_step"] // self.config.early_stopping.validate_every
            self.running_values["prev_epoch_time"] = time.time() - self.start
            self.start = time.time()
            self.model.train()

        for mean_key in self.running_values["val_loss"]: pbar_message += " val {}: {:.3f} ".format(mean_key, self.running_values["val_loss"][mean_key])

        pbar_message += " time {:.1f} sec/step, ".format(self.running_values["prev_epoch_time"])
        pbar_message += "saved at {}".format(self.running_values["saved_at_step"])
        return pbar_message


    def monitoring(self, train_metrics, val_metrics):

        self._find_learning_rate()

        self._update_train_val_logs(train_metrics = train_metrics, val_metrics = val_metrics)
        # wandb.log({"train": train_metrics, "val": val_metrics})

        #Flag if its saved dont save it again on $save_every
        not_saved = True

        #If we have a better validation loss
        if (val_metrics["val_loss"]["total"] < self.logs["best_logs"]["val_loss"]["total"]):
            self._update_best_logs(current_step = self.logs["current_step"], val_metrics = val_metrics)
            self.best_model.load_state_dict(self.model.state_dict())
            if self.config.training_params.rec_test:
                self._test_n_update()

            self.logs["saved_step"] = self.logs["current_step"]
            self.logs["steps_no_improve"] = 0
            self.save_model_logs(self.config.model.save_dir, verbose = False)
            not_saved = False
        else:
            self.logs["steps_no_improve"] += 1
            if self.config.training_params.rec_test and self.config.training_params.test_on_bottoms:
                self._test_n_update()

        return self._early_stop_check_n_save(not_saved)


    def checkpointing(self, batch_loss, predictions, targets):

        targets_tens = torch.cat(targets).cpu().numpy().flatten()
        target_dict = { pred_key: targets_tens for pred_key in predictions[0]}

        total_preds, train_metrics  = {}, defaultdict(dict)
        train_metrics["train_loss"] = dict(batch_loss)
        for pred_key in predictions[0]:
            total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in predictions if pred_key in pred],axis=0).argmax(axis=-1)
            train_metrics["train_acc"][pred_key] =  np.equal(target_dict[pred_key], total_preds[pred_key]).sum() / len(target_dict[pred_key])
            train_metrics["train_f1"][pred_key] = f1_score(total_preds[pred_key], target_dict[pred_key], average="macro")
            train_metrics["train_k"][pred_key] = cohen_kappa_score(total_preds[pred_key], target_dict[pred_key])
            train_metrics["train_perclassf1"][pred_key] = f1_score(total_preds[pred_key], target_dict[pred_key], average=None)

        train_metrics = dict(train_metrics) #Avoid passing empty dicts to logs, better return an error!

        val_metrics = self.validate()
        early_stop = self.monitoring(train_metrics=train_metrics, val_metrics=val_metrics)
        return early_stop, val_metrics["val_loss"]



    def _update_best_logs(self, current_step, val_metrics):

        val_metrics.update({"step": current_step})
        self.logs["best_logs"] = val_metrics

        if self.config.training_params.verbose:
            step = int(current_step / self.config.early_stopping.validate_every)
            if not self.config.training_params.tdqm_disable: print()

            message = Fore.WHITE + "Epoch {0:d} step {1:d} with ".format(self.logs["current_epoch"], step)
            if "val_loss" in val_metrics:
                for i, v in val_metrics["val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
            if "val_acc" in val_metrics:
                for i, v in val_metrics["val_acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i,
                                                                                                                    v * 100)
            if "val_f1" in val_metrics:
                for i, v in val_metrics["val_f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i,
                                                                                                                   v * 100)
            if "val_k" in val_metrics:
                for i, v in val_metrics["val_k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
            # if "val_perclassf1" in val_metrics:
            #     for i, v in val_metrics["val_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((v*100).round(2)))))
            print(message)
    def _test_n_update(self):
        val_metrics = self.test()

        self.logs["test_logs"][self.logs["current_step"]] = val_metrics

        if self.config.training_params.verbose:
            message = Fore.WHITE + "Test "
            if "test_loss" in val_metrics:
                for i, v in val_metrics["test_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
            if "test_acc" in val_metrics:
                for i, v in val_metrics["test_acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i,
                                                                                                                     v * 100)
            if "test_f1" in val_metrics:
                for i, v in val_metrics["test_f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i,
                                                                                                                   v * 100)
            if "test_k" in val_metrics:
                for i, v in val_metrics["test_k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
            if "test_perclassf1" in val_metrics:
                for i, v in val_metrics["test_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(
                    i,
                    "{}".format(
                        str(list(
                            (
                                    v * 100).round(
                                2)))))
            print(message)
    def _early_stop_check_n_save(self, not_saved):

        training_cycle = (self.logs["current_step"] // self.config.early_stopping.validate_every)
        if not_saved and training_cycle % self.config.early_stopping.save_every == 0:
            # Some epochs without improvement have passed, we save to avoid losing progress even if its not giving new best
            self.save_model_logs(self.config.model.save_dir)
            self.logs["saved_step"] = self.logs["current_step"]

        if training_cycle == self.config.early_stopping.n_steps_stop_after:
            # After 'n_steps_stop_after' we need to start counting till we reach the earlystop_threshold
            self.steps_at_earlystop_threshold = self.logs["steps_no_improve"] # we dont need to initialize that since training_cycle > self.agent.config.n_steps_stop_after will not be true before ==

        early_stop = False
        # if "steps_at_earlystop_threshold" not in self.keys(): self.steps_at_earlystop_threshold = 150 #TODO: save this in logs or eliminate it
        if training_cycle > self.config.early_stopping.n_steps_stop_after and self.logs["steps_no_improve"] >= self.config.early_stopping.n_steps_stop:
            early_stop = True
        return early_stop

    def _find_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']

    def _update_train_val_logs(self, train_metrics, val_metrics):

        train_metrics.update({  "validate_every": self.config.early_stopping.validate_every,
                                "batch_size": self.config.training_params.batch_size,
                                "learning_rate": self.scheduler.lr_history[
                                                  max(self.logs["current_step"] - self.config.early_stopping.validate_every, 0):
                                                  self.logs["current_step"]]})

        self.logs["val_logs"][self.logs["current_step"]] = val_metrics
        self.logs["train_logs"][self.logs["current_step"]] = train_metrics

    def _calc_mean_batch_loss(self, batch_loss):
        mean_batch = defaultdict(list)
        for b_i in batch_loss:
            for loss_key in b_i:
                mean_batch[loss_key].append(b_i[loss_key])
        for key in mean_batch:
            mean_batch[key] = np.array(mean_batch[key]).mean(axis=0)
        return mean_batch

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose = True):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params =  sum(p.numel() for p in unique)
        if verbose:
            print("Total number of trainable parameters are: {}".format(model_total_params))

        return model_total_params