from torch import nn
from typing import Any, Dict, List, Tuple, Union, Callable
from enum import Enum
from fate.arch import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments as _hf_TrainingArguments, PreTrainedTokenizer
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import has_length
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import _utils
from fate.ml.aggregator.base import Aggregator
import logging
from transformers import logging as transformers_logging
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from typing import Optional
from dataclasses import dataclass, field, fields
from fate.ml.aggregator import AggregatorType
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from transformers.trainer import logger as logger_
from fate.ml.evaluation.metric_base import MetricEnsemble
from transformers import IntervalStrategy, DefaultFlowCallback
from torch.nn import Module
from fate.ml.nn.trainer.trainer_base import HomoTrainerClient, HomoTrainerServer
from fate.ml.nn.homo.fedavg import FedArguments, FedAVGArguments, FedAVGClient, FedAVGServer, TrainingArguments
from fate.ml.aggregator import AggregatorClientWrapper, AggregatorServerWrapper

logger = logging.getLogger(__name__)

class AggregateStrategy(Enum):
    EPOCH = "epoch"
    STEP = "steps"


class TrainerClient(HomoTrainerClient):
    def __init__(
        self,
        ctx: Context,
        model: Module,
        training_args: TrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: Module = None,
        optimizer: Optimizer = None,
        scheduler: _LRScheduler = None,
        callbacks: List[TrainerCallback] = [],
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        use_hf_default_behavior: bool = False,
        compute_metrics: Callable = None,
        local_mode: bool = False,
    ):
        super().__init__(
            ctx,
            model,
            training_args,
            fed_args,
            train_set,
            val_set,
            loss_fn,
            optimizer,
            data_collator,
            scheduler,
            tokenizer,
            callbacks,
            use_hf_default_behavior,
            compute_metrics=compute_metrics,
            local_mode=local_mode,
        )

    def init_aggregator(self, ctx: Context, fed_args: FedArguments):
        print("no aggregator")
        # aggregate_type = "weighted_mean"
        # aggregator_name = "fedavg"
        # aggregator = fed_args.aggregator
        # return AggregatorClientWrapper(
        #     ctx, aggregate_type, aggregator_name, aggregator, sample_num=len(self.train_dataset), args=self._args
        # )

    def on_federation(
        self,
        ctx: Context,
        aggregator: AggregatorClientWrapper,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        self.model.fedavg()

    # def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if self.wrapped_trainer.local_mode:
    #         return
    #     if self.fed_arg.aggregate_strategy == AggregateStrategy.EPOCH.value:
    #         if self.wrapped_trainer.aggregation_checker.should_aggregate(state):
    #             logger.info("my epoch end")
                # agg_round = self.wrapped_trainer.aggregation_checker.model_aggregation_count
                # sub_ctx = self.ctx.sub_ctx("aggregation").indexed_ctx(agg_round)
                # ret = self._call_wrapped(
                #     sub_ctx,
                #     self.wrapped_trainer.aggregator,
                #     self.fed_arg,
                #     "on_federation",
                #     args=args,
                #     state=state,
                #     control=control,
                #     **kwargs,
                # )
                # self.wrapped_trainer.aggregation_checker.inc_model_agg_count()
                # return ret


class TrainerServer(HomoTrainerServer):
    def __init__(self, ctx: Context, local_mode: bool = False) -> None:
        super().__init__(ctx, local_mode)

    def init_aggregator(self, ctx):
        return AggregatorServerWrapper(ctx)

    def on_federation(self, ctx: Context, aggregator: AggregatorServerWrapper, agg_iter_idx: int):
        aggregator.model_aggregation(ctx)

    def train(self):
        if self.local_mode:
            logger.info("Local model is set, skip initializing fed setting & aggregator")
            return

        self.aggregator: Aggregator = self.init_aggregator(self.ctx)
        logger.info("Initialized aggregator Done: {}".format(self.aggregator))
        self._parameter_check_callback.on_train_begin(None, None, None)  # only get parameters from clients and align
        parameters = self._parameter_check_callback.get_parameters()
        self._max_aggregation = parameters["max_aggregation"]
        self.can_aggregate_loss = parameters["can_aggregate_loss"]
        logger.info("checked parameters are {}".format(parameters))

        self.on_init_end(self.ctx, aggregator=self.aggregator)
        self.on_train_begin(self.ctx, aggregator=self.aggregator)

        ctx = self.ctx
        for i in range(self._max_aggregation):
            sub_ctx = ctx.sub_ctx("aggregation").indexed_ctx(i)
            self.on_federation(sub_ctx, aggregator=self.aggregator, agg_iter_idx=i)
            # if self.can_aggregate_loss:
            #     loss_sub_ctx = ctx.sub_ctx("loss_aggregation").indexed_ctx(i)
            #     loss = self.aggregator.loss_aggregation(loss_sub_ctx)
            #     sub_ctx.metrics.log_loss("loss", loss)

        self.on_train_end(self.ctx, aggregator=self.aggregator)