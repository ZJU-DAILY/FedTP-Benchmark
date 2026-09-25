import torch
from transformers import Trainer
from collections.abc import Mapping

from fate.ml.nn.homo.fedavg import FedAVGClient
from lib.utils import align_prediction_and_target, unpack_spatiotemporal_batch
from privacy.attack_trace import capture_revised_quantized_prediction


def _unwrap_inputs(inputs):
    current = inputs
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    return current


def _parse_xy_from_inputs(inputs):
    inputs = _unwrap_inputs(inputs)
    # FATE versions differ here: some collators wrap the usual mapping in a
    # tuple/list, others return the raw (x, y) sequence.  Never call .get on
    # an arbitrary sequence.
    if isinstance(inputs, Mapping):
        if "x" in inputs and "labels" in inputs:
            return inputs["x"], inputs["labels"]
        x = inputs.get("input_ids", inputs.get("inputs"))
        y = inputs.get("labels", inputs.get("y", inputs.get("targets")))
        return x, y
    if isinstance(inputs, (list, tuple)) and len(inputs) == 1 and isinstance(inputs[0], Mapping):
        return _parse_xy_from_inputs(inputs[0])
    return unpack_spatiotemporal_batch(inputs)


def _move_to_model_device(x, y, model):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return x, y

    if hasattr(x, "to"):
        x = x.to(device)
    if hasattr(y, "to"):
        y = y.to(device)
    return x, y


class FedGRUTrainer(Trainer):
    def __init__(self, *args, loss_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        x, y = _parse_xy_from_inputs(inputs)
        x, y = _move_to_model_device(x, y, model)

        if x is None or y is None:
            raise ValueError("FedGRUTrainer could not parse (x, y) from the batch inputs.")

        pred = model(x)
        trace_args = getattr(self, "privacy_args", None)
        trace_ctx = getattr(self, "ctx", None)
        if trace_args is not None and trace_ctx is not None:
            capture_revised_quantized_prediction(
                trace_ctx, trace_args, "fedgru_prediction",
                prediction=pred, model_state_dict=model.state_dict(),
            )
        pred, y = align_prediction_and_target(pred, y)
        loss = self.loss_func(pred, y)
        return (loss, pred) if return_outputs else loss


class FedGRUFedAVGClient(FedAVGClient):
    def __init__(
        self,
        ctx,
        model,
        train_set,
        val_set,
        optimizer,
        loss_fn,
        scheduler,
        training_args,
        fed_args,
        compute_metrics=None,
        **kwargs,
    ):
        super().__init__(
            ctx=ctx,
            model=model,
            train_set=train_set,
            val_set=val_set,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            training_args=training_args,
            fed_args=fed_args,
            compute_metrics=compute_metrics,
            **kwargs,
        )
        self.loss_fn = loss_fn
        self.loss_func = loss_fn

        self.trainer = FedGRUTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),
            loss_func=loss_fn,
        )

        if hasattr(loss_fn, "to"):
            self.trainer.loss_func = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        x, y = _parse_xy_from_inputs(inputs)
        x, y = _move_to_model_device(x, y, model)

        if x is None or y is None:
            raise ValueError("FedGRUFedAVGClient could not parse (x, y) from batch inputs.")

        pred = model(x)
        pred, y = align_prediction_and_target(pred, y)
        loss = self.loss_fn(pred, y)
        return (loss, pred) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        x, y = _parse_xy_from_inputs(inputs)
        x, y = _move_to_model_device(x, y, model)
        if x is None or y is None:
            raise ValueError("FedGRUFedAVGClient could not parse (x, y) during prediction_step.")

        with torch.no_grad():
            pred = model(x)
            pred, y = align_prediction_and_target(pred, y)
            loss = self.loss_fn(pred, y)

        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None
        return loss, pred.detach(), y.detach()
