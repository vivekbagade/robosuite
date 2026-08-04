import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from policy import ACTPolicy
from copy import deepcopy
from utils.utils import *

device = os.environ['DEVICE']

class ACTTrainer:
    def __init__(self, policy_config, train_cfg, base_weights_dir, checkpoint_dir):
        self.policy_config = policy_config
        self.train_cfg = train_cfg
        self.base_weights_dir = base_weights_dir
        self.checkpoint_dir = checkpoint_dir

    def make_policy(self):
        """Construct the policy to train.

        Extracted so variants (see ``train_waypoint.WaypointACTTrainer``) can
        swap the policy class without duplicating ``train_bc``.
        """
        return ACTPolicy(self.policy_config)

    def make_optimizer(self, policy_class, policy):
        if policy_class in ['ACT', 'CNNMLP']:
            optimizer = policy.configure_optimizers()
        else:
            raise ValueError(f"Unknown policy class: {policy_class}")
        return optimizer

    def forward_pass(self, data, policy):
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
        return policy(qpos_data, image_data, action_data, is_pad)

    def plot_history(self, train_history, validation_history, num_epochs, ckpt_dir, seed):
        # save training curves
        for key in train_history[0]:
            plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
            plt.figure()
            
            # Since validation_history is per epoch, its length is the number of epochs
            num_epochs_completed = len(validation_history)
            if num_epochs_completed == 0:
                continue

            # train_history is per batch, so we need to average it per epoch
            num_train_batches = len(train_history) // num_epochs_completed
            train_values_per_epoch = []
            if num_train_batches > 0:
                for i in range(num_epochs_completed):
                    epoch_train_history = train_history[i*num_train_batches:(i+1)*num_train_batches]
                    epoch_train_values = [summary[key].item() for summary in epoch_train_history]
                    train_values_per_epoch.append(np.mean(epoch_train_values))

            val_epochs = np.arange(num_epochs_completed)
            val_values = [summary[key].item() for summary in validation_history]
            
            # Plot train loss per epoch
            if train_values_per_epoch:
                plt.plot(val_epochs, train_values_per_epoch, label='train')
            plt.plot(val_epochs, val_values, label='validation')
            
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
            plt.close() # Close the figure to free memory
        print(f'Saved plots to {ckpt_dir}')

    def train_bc(self, train_dataloader, val_dataloader):
        # load policy
        ckpt_path = None
        if self.base_weights_dir is not None:
            ckpt_path = os.path.join(self.base_weights_dir, self.train_cfg['eval_ckpt_name'])
            print(f"Checkpoint path: {ckpt_path}")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist. Please check the path.")
        policy = self.make_policy()
        if ckpt_path is not None:
            loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
            print(f"Loading status: {loading_status}")
        policy.to(device)

        # load optimizer
        optimizer = self.make_optimizer(self.policy_config['policy_class'], policy)

        # create checkpoint dir if not exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None
        for epoch in range(self.train_cfg['num_epochs']):
            print(f'\nEpoch {epoch}')
            # validation
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for _, data in enumerate(val_dataloader):
                    forward_dict = self.forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            # training
            policy.train()
            optimizer.zero_grad()
            epoch_start = len(train_history)
            for _, data in enumerate(train_dataloader):
                forward_dict = self.forward_pass(data, policy)
                # backward
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            epoch_summary = compute_dict_mean(train_history[epoch_start:])
            epoch_train_loss = epoch_summary['loss']
            print(f'Train loss: {epoch_train_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            if (epoch + 1) % 100 == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"policy_epoch_{epoch}_seed_{self.train_cfg['seed']}.ckpt")
                torch.save(policy.state_dict(), ckpt_path)
                self.plot_history(train_history, validation_history, epoch, self.checkpoint_dir, self.train_cfg['seed'])

        ckpt_path = os.path.join(self.checkpoint_dir, f'policy_last.ckpt')
        torch.save(policy.state_dict(), ckpt_path)
        if best_ckpt_info is not None:
            best_epoch, best_val_loss, best_state_dict = best_ckpt_info
            best_ckpt_path = os.path.join(self.checkpoint_dir, 'policy_best.ckpt')
            torch.save(best_state_dict, best_ckpt_path)
            print(f'Best checkpoint (epoch {best_epoch}, val loss {best_val_loss:.5f}) saved to {best_ckpt_path}')