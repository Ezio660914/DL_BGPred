# -*- coding: utf-8 -*-
import os
import sys
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TimeSeriesPrediction(pl.LightningModule):
    def __init__(
            self,
            model=None
    ):
        super().__init__()
        self.automatic_optimization = False
        self.model = model

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def _process_train_inputs(self, inputs):
        inputs[:, :, 0] = (inputs[:, :, 0] - 40) / (400 - 40)  # bg
        inputs[:, :, 1] = inputs[:, :, 1] / 20  # bolus
        inputs[:, :, 2] = inputs[:, :, 2] / 200  # carbs
        inputs[:, :, 3] = inputs[:, :, 3] / 287  # ToD
        inputs = inputs[:, :, :self.model.input_channels]
        return inputs

    def _process_val_inputs(self, inputs):
        return self._process_train_inputs(inputs)

    def _process_outputs(self, bg_recover):
        bg_recover = bg_recover * (400 - 40) + 40
        return bg_recover

    def _get_loss(self, *args, **kwargs):
        raise NotImplementedError

    def general_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        step_output = self.general_step(batch, batch_idx)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(step_output['loss'])
        opt.step()
        return step_output

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.general_step(batch, batch_idx)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.general_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)
