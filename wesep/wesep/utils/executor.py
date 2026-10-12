# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2026 Ke Zhang (kylezhang1118@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import tableprint as tp
import torch

from wesep.utils.funcs import clip_gradients
from wesep.dataset.collate import AUX_KEY_MAP


class Executor:

    def __init__(self, aux_key_map=None):
        self.step = 0
        self.aux_key_map = aux_key_map or AUX_KEY_MAP

        # cue
        self.cue_keys = list(self.aux_key_map.values())

    # -------------------------
    # helpers
    # -------------------------

    def _extract_model_inputs(self, batch, device):
        """
        Build model inputs from collated batch.

        Args:
            batch: dict from tse_collate_fn
            device: torch.device

        Returns:
            mix:    Tensor [B, 1, T]
            cues:   list[Tensor] or None
            target: Tensor [B, 1, T]
        """
        if "wav_mix" not in batch:
            raise RuntimeError("[executor] Missing required key: wav_mix")
        if "wav_target" not in batch:
            raise RuntimeError("[executor] Missing required key: wav_target")

        mix = batch["wav_mix"].float().to(device)
        target = batch["wav_target"].float().to(device)

        cues = []
        for k in self.cue_keys:
            if k in batch and batch[k] is not None:
                cues.append(batch[k].float().to(device))

        if len(cues) == 0:
            cues = None

        return mix, cues, target

    # -------------------------
    # train
    # -------------------------

    def train(self,
              dataloader,
              models,
              epoch_iter,
              optimizers,
              criterion,
              schedulers,
              scaler,
              epoch,
              enable_amp,
              logger,
              clip_grad=5.0,
              log_batch_interval=100,
              device=torch.device("cuda"),
              se_loss_weight=1.0):

        model = models[0]
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        model.train()
        log_interval = log_batch_interval
        losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                mix, cues, target = self._extract_model_inputs(batch, device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    # ---- forward ----
                    if cues is None:
                        outputs = model(mix)
                    else:
                        outputs = model(mix, cues)

                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]

                    # ---- loss ----
                    loss = 0.0
                    for ii in range(len(criterion)):
                        for ji in range(len(se_loss_weight[0][ii])):
                            out_idx = se_loss_weight[0][ii][ji]
                            w = se_loss_weight[1][ii][ji]
                            loss = loss + w * (criterion[ii](outputs[out_idx],
                                                             target).mean())

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                # ---- backward ----
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            (
                                "TRAIN",
                                epoch,
                                i + 1,
                                total_loss_avg,
                                optimizer.param_groups[0]["lr"],
                            ),
                            width=10,
                            style="grid",
                        ))

                if (i + 1) == epoch_iter:
                    break

        total_loss_avg = sum(losses) / len(losses)
        return total_loss_avg, 0

    # -------------------------
    # cv / validation
    # -------------------------

    def cv(self,
           dataloader,
           models,
           val_iter,
           criterion,
           epoch,
           enable_amp,
           logger,
           log_batch_interval=100,
           device=torch.device("cuda")):

        model = models[0]
        model.eval()

        log_interval = log_batch_interval
        losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                mix, cues, target = self._extract_model_inputs(batch, device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    if cues is None:
                        outputs = model(mix)
                    else:
                        outputs = model(mix, cues)

                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]

                    # 默认第一个 loss 作为验证指标
                    loss = criterion[0](outputs[0], target).mean()

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, total_loss_avg, "-"),
                            width=10,
                            style="grid",
                        ))

                if (i + 1) == val_iter:
                    break

        total_loss_avg = sum(losses) / len(losses)
        return total_loss_avg, 0