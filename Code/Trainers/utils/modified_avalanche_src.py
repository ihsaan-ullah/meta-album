"""
This module contains a modified version of avalanche source code (version 0.1.0)
These modifications are done to allow more customization or integrate future avalanche version.

https://github.com/ContinualAI/avalanche/releases/tag/v0.1.0

Copyright (c) 2020 ContinualAI.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import numpy as np
import warnings
from collections import defaultdict
from fnmatch import fnmatch

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _NormBase

from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

from avalanche.training.plugins.evaluation import EvaluationPlugin, accuracy_metrics
from avalanche.training.plugins.evaluation import loss_metrics, InteractiveLogger
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.utils import get_layers_and_params
from avalanche.models import MultiTaskModule


default_evaluator = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True,
                     experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()],
    suppress_warnings=True,
)

class PNNStrategy(BaseStrategy):
    """Progressive Neural Network strategy.
    To use this strategy you need to instantiate a PNN model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins=None,
        evaluator=default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param **base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        # The only thing this strategy class differs from the Naive one is 
        # that it checks whether the model has the correct architecture.
        from avalanche.models.pnn import PNN
        assert isinstance(model, PNN), "PNNStrategy requires a PNN model."
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class EWCPlugin(StrategyPlugin):
    """
    Elastic Weight Consolidation (EWC) plugin.
    EWC computes importance of each weight at the end of training on current
    experience. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on previous experiences in proportion to their importance
    on that experience. Importances are computed with an additional pass on the
    training set. This plugin does not use task identities.
    """

    def __init__(
        self,
        ewc_lambda,
        mode="separate",
        decay_factor=None,
        keep_importance_data=False,
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param decay_factor: used only if mode is `online`.
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """
        
        if mode == "online":
            warnings.warn(
                """The current implementation of Online EWC does not work when  
                the multi-head model has different number of classes for each head, 
                consider using mode="separate" (ewc_mode) """
            )

        super().__init__()
        assert (decay_factor is None) or (
            mode == "online"
        ), "You need to set `online` mode to use `decay_factor`."
        assert (decay_factor is not None) or (
            mode != "online"
        ), "You need to set `decay_factor` to use the `online` mode."
        assert (
            mode == "separate" or mode == "online"
        ), "Mode must be separate or online."

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == "separate":
            for experience in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[experience],
                    self.importances[experience],
                ):
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = saved_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == "online":
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                strategy.model.named_parameters(),
                self.saved_params[prev_exp],
                self.importances[prev_exp],
            ):
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = saved_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in zip(
                self.importances[t - 1], importances
            ):
                assert k1 == k2, "Error in importance computation."
                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


class EWC(BaseStrategy):
    """Elastic Weight Consolidation (EWC) strategy.
    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor = None,
        keep_importance_data: bool = False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins = None,
        evaluator = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.
        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param **base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        

class SynapticIntelligencePlugin(StrategyPlugin):
    """
    The Synaptic Intelligence plugin.
    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)
    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).
    This plugin can be attached to existing strategies to achieve a
    regularization effect.
    This plugin will require the strategy `loss` field to be set before the
    `before_backward` callback is invoked. The loss Tensor will be updated to
    achieve the S.I. regularization effect.
    """

    def __init__(
        self,
        si_lambda,
        eps: float = 0.0000001,
        excluded_parameters = None,
        device = "as_strategy",
    ):
        """
        Creates an instance of the Synaptic Intelligence plugin.
        :param si_lambda: Synaptic Intelligence lambda term.
            If list, one lambda for each experience. If the list has less
            elements than the number of experiences, last lambda will be
            used for the remaining experiences.
        :param eps: Synaptic Intelligence damping parameter.
        :param device: The device to use to run the S.I. experiences.
            Defaults to "as_strategy", which means that the `device` field of
            the strategy will be used. Using a different device may lead to a
            performance drop due to the required data transfer.
        """

        super().__init__()

        warnings.warn(
            "The Synaptic Intelligence plugin is in an alpha stage "
            "and is not perfectly aligned with the paper "
            "implementation. Please use at your own risk!"
        )

        if excluded_parameters is None:
            excluded_parameters = []
        self.si_lambda = (
            si_lambda if isinstance(si_lambda, (list, tuple)) else [si_lambda]
        )
        self.eps: float = eps
        self.excluded_parameters = set(excluded_parameters)
        self.ewc_data = (dict(), dict())
        """
        The first dictionary contains the params at loss minimum while the 
        second one contains the parameter importance.
        """

        self.syn_data = {
            "old_theta": dict(),
            "new_theta": dict(),
            "grad": dict(),
            "trajectory": dict(),
            "cum_trajectory": dict(),
        }

        self._device = device

    def before_training_exp(self, strategy, **kwargs):
        super().before_training_exp(strategy, **kwargs)
        SynapticIntelligencePlugin.create_syn_data(
            strategy.model,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
        )

        SynapticIntelligencePlugin.init_batch(
            strategy.model,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
        )

    def before_backward(self, strategy, **kwargs):
        super().before_backward(strategy, **kwargs)

        exp_id = strategy.clock.train_exp_counter
        try:
            si_lamb = self.si_lambda[exp_id]
        except IndexError:  # less than one lambda per experience, take last
            si_lamb = self.si_lambda[-1]

        syn_loss = SynapticIntelligencePlugin.compute_ewc_loss(
            strategy.model,
            self.ewc_data,
            self.excluded_parameters,
            lambd=si_lamb,
            device=self.device(strategy),
        )

        if syn_loss is not None:
            strategy.loss += syn_loss.to(strategy.device)

    def before_training_iteration(
        self, strategy, **kwargs
    ):
        super().before_training_iteration(strategy, **kwargs)
        SynapticIntelligencePlugin.pre_update(
            strategy.model, self.syn_data, self.excluded_parameters
        )

    def after_training_iteration(
        self, strategy, **kwargs
    ):
        super().after_training_iteration(strategy, **kwargs)
        SynapticIntelligencePlugin.post_update(
            strategy.model, self.syn_data, self.excluded_parameters
        )

    def after_training_exp(self, strategy, **kwargs):
        super().after_training_exp(strategy, **kwargs)
        SynapticIntelligencePlugin.update_ewc_data(
            strategy.model,
            self.ewc_data,
            self.syn_data,
            0.001,
            self.excluded_parameters,
            1,
            eps=self.eps,
        )

    def device(self, strategy):
        if self._device == "as_strategy":
            return strategy.device

        return self._device

    @staticmethod
    @torch.no_grad()
    def create_syn_data(
        model,
        ewc_data,
        syn_data,
        excluded_parameters,
    ):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        for param_name, param in params:
            if param_name in ewc_data[0]:
                continue

            # Handles added parameters (doesn't manage parameter expansion!)
            ewc_data[0][param_name] = SynapticIntelligencePlugin._zero(param)
            ewc_data[1][param_name] = SynapticIntelligencePlugin._zero(param)

            syn_data["old_theta"][
                param_name
            ] = SynapticIntelligencePlugin._zero(param)
            syn_data["new_theta"][
                param_name
            ] = SynapticIntelligencePlugin._zero(param)
            syn_data["grad"][param_name] = SynapticIntelligencePlugin._zero(
                param
            )
            syn_data["trajectory"][
                param_name
            ] = SynapticIntelligencePlugin._zero(param)
            syn_data["cum_trajectory"][
                param_name
            ] = SynapticIntelligencePlugin._zero(param)

    @staticmethod
    @torch.no_grad()
    def _zero(param):
        return torch.zeros(param.numel(), dtype=param.dtype)

    @staticmethod
    @torch.no_grad()
    def extract_weights(
        model, target, excluded_parameters
    ):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        for name, param in params:
            target[name][...] = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target, excluded_parameters):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        # Store the gradients into target
        for name, param in params:
            target[name][...] = param.grad.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def init_batch(
        model,
        ewc_data,
        syn_data,
        excluded_parameters,
    ):
        # Keep initial weights
        SynapticIntelligencePlugin.extract_weights(
            model, ewc_data[0], excluded_parameters
        )
        for param_name, param_trajectory in syn_data["trajectory"].items():
            param_trajectory.fill_(0.0)

    @staticmethod
    @torch.no_grad()
    def pre_update(model, syn_data, excluded_parameters):
        SynapticIntelligencePlugin.extract_weights(
            model, syn_data["old_theta"], excluded_parameters
        )

    @staticmethod
    @torch.no_grad()
    def post_update(
        model, syn_data, excluded_parameters
    ):
        SynapticIntelligencePlugin.extract_weights(
            model, syn_data["new_theta"], excluded_parameters
        )
        SynapticIntelligencePlugin.extract_grad(
            model, syn_data["grad"], excluded_parameters
        )

        for param_name in syn_data["trajectory"]:
            syn_data["trajectory"][param_name] += syn_data["grad"][
                param_name
            ] * (
                syn_data["new_theta"][param_name]
                - syn_data["old_theta"][param_name]
            )

    @staticmethod
    def compute_ewc_loss(
        model,
        ewc_data,
        excluded_parameters,
        device,
        lambd=0.0,
    ):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        loss = None
        for name, param in params:
            weights = param.to(device).flatten()  # Flat, not detached
            param_ewc_data_0 = ewc_data[0][name].to(device)  # Flat, detached
            param_ewc_data_1 = ewc_data[1][name].to(device)  # Flat, detached

            syn_loss = torch.dot(
                param_ewc_data_1, (weights - param_ewc_data_0) ** 2
            ) * (lambd / 2)

            if loss is None:
                loss = syn_loss
            else:
                loss += syn_loss

        return loss

    @staticmethod
    @torch.no_grad()
    def update_ewc_data(
        net,
        ewc_data,
        syn_data,
        clip_to: float,
        excluded_parameters,
        c=0.0015,
        eps: float = 0.0000001,
    ):
        SynapticIntelligencePlugin.extract_weights(
            net, syn_data["new_theta"], excluded_parameters
        )

        for param_name in syn_data["cum_trajectory"]:
            syn_data["cum_trajectory"][param_name] += (
                c
                * syn_data["trajectory"][param_name]
                / (
                    np.square(
                        syn_data["new_theta"][param_name]
                        - ewc_data[0][param_name]
                    )
                    + eps
                )
            )

        for param_name in syn_data["cum_trajectory"]:
            ewc_data[1][param_name] = torch.empty_like(
                syn_data["cum_trajectory"][param_name]
            ).copy_(-syn_data["cum_trajectory"][param_name])

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in ewc_data[1]:
            ewc_data[1][param_name] = torch.clamp(
                ewc_data[1][param_name], max=clip_to
            )
            ewc_data[0][param_name] = syn_data["new_theta"][param_name].clone()

    @staticmethod
    def explode_excluded_parameters(excluded):
        """
        Explodes a list of excluded parameters by adding a generic final ".*"
        wildcard at its end.
        :param excluded: The original set of excluded parameters.
        :return: The set of excluded parameters in which ".*" patterns have been
            added.
        """
        result = set()
        for x in excluded:
            result.add(x)
            if not x.endswith("*"):
                result.add(x + ".*")
        return result

    @staticmethod
    def not_excluded_parameters(
        model, excluded_parameters
    ):
        # Add wildcards ".*" to all excluded parameter names
        result = []
        excluded_parameters = (
            SynapticIntelligencePlugin.explode_excluded_parameters(
                excluded_parameters
            )
        )
        layers_params = get_layers_and_params(model)

        for lp in layers_params:
            if isinstance(lp.layer, _NormBase):
                # Exclude batch norm parameters
                excluded_parameters.add(lp.parameter_name)

        for name, param in model.named_parameters():
            accepted = True
            for exclusion_pattern in excluded_parameters:
                if fnmatch(name, exclusion_pattern):
                    accepted = False
                    break

            if accepted:
                result.append((name, param))

        return result

    @staticmethod
    def allowed_parameters(
        model, excluded_parameters
    ):

        allow_list = SynapticIntelligencePlugin.not_excluded_parameters(
            model, excluded_parameters
        )

        result = []
        for name, param in allow_list:
            if param.requires_grad:
                result.append((name, param))

        return result


class SynapticIntelligence(BaseStrategy):
    """Synaptic Intelligence strategy.
    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)
    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).
    The Synaptic Intelligence regularization can also be used in a different
    strategy by applying the :class:`SynapticIntelligencePlugin` plugin.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        si_lambda,
        eps: float = 0.0000001,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins = None,
        evaluator=default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.
        Creates an instance of the Synaptic Intelligence strategy.
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param si_lambda: Synaptic Intelligence lambda term.
            If list, one lambda for each experience. If the list has less
            elements than the number of experiences, last lambda will be
            used for the remaining experiences.
        :param eps: Synaptic Intelligence damping parameter.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param **base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        if plugins is None:
            plugins = []

        # This implementation relies on the S.I. Plugin, which contains the
        # entire implementation of the strategy!
        plugins.append(SynapticIntelligencePlugin(
            si_lambda=si_lambda, eps=eps))

        super(SynapticIntelligence, self).__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class LwFPlugin(StrategyPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None

        self.prev_classes = {'0': set()}
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """

    def _distillation_loss(self, out, prev_out, active_units):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)
        log_p = torch.log_softmax(out / self.temperature, dim=1)[:, au]
        q = torch.softmax(prev_out / self.temperature, dim=1)[:, au]
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    def penalty(self, out, x, alpha, curr_model):
        """
        Compute weighted distillation loss.
        """
        if self.prev_model is None:
            return 0
        else:
            with torch.no_grad():
                if isinstance(self.prev_model, MultiTaskModule):
                    # output from previous output heads.
                    y_prev = avalanche_forward(self.prev_model, x, None)
                    # in a multitask scenario we need to compute the output
                    # from all the heads, so we need to call forward again.
                    y_curr = avalanche_forward(curr_model, x, None)
                else:  # no task labels
                    y_prev = {'0': self.prev_model(x)}
                    y_curr = {'0': out}

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads.
                if str(task_id) in self.prev_classes:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    au = self.prev_classes[str(task_id)]
                    dist_loss += self._distillation_loss(yc, yp, au)
            
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        alpha = self.alpha[strategy.clock.train_exp_counter] \
            if isinstance(self.alpha, (list, tuple)) else self.alpha
        penalty = self.penalty(strategy.mb_output, strategy.mb_x, alpha,
                               strategy.model)
        strategy.loss += penalty
    
    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.prev_model = copy.deepcopy(strategy.model)
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[task_id]\
                    .union(pc)



class LwF(BaseStrategy):
    """ Learning without Forgetting (LwF) strategy.

    See LwF plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer, criterion,
                 alpha, temperature: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins = None,
                 evaluator = default_evaluator, eval_every=-1):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        lwf = LwFPlugin(alpha, temperature)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)








