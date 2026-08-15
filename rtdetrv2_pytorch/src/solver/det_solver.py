"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def _new_distant_accumulator(self):
        """A fresh distant-target accumulator per epoch, or None when not configured.

        Opt-in through `distant_target_metric` in the config, so every existing model is untouched:

            distant_target_metric:
              width_range: [4, 12]     # apparent width in source px, the contested band
              iou_threshold: 0.3       # loose: finding it matters, the tracker refines it
              score_threshold: 0.25    # must match the deployed operating point
        """
        config = self.cfg.yaml_cfg.get('distant_target_metric')
        if not config:
            return None

        from ..misc.distant_target_metric import (
            CONTESTED_WIDTH_PX,
            DEFAULT_IOU,
            DEFAULT_SCORE,
            DistantTargetAccumulator,
        )

        return DistantTargetAccumulator(
            self.val_dataloader.dataset.coco,
            width_range=config.get('width_range', CONTESTED_WIDTH_PX),
            iou_threshold=config.get('iou_threshold', DEFAULT_IOU),
            score_threshold=config.get('score_threshold', DEFAULT_SCORE),
        )

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        with open(str(self.output_dir / 'config.txt'), 'w') as f:
            f.write(str(self.cfg.__dict__))

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device,
                distant_accumulator=self._new_distant_accumulator(),
            )

            # TODO 
            # Which metric decides best.pth. Defaults to COCO AP, so existing configs are
            # unchanged. A band model can set `checkpoint_metric: distant_target_recall` instead,
            # because aggregate AP is not what it is judged on.
            selection_metric = self.cfg.yaml_cfg.get('checkpoint_metric', 'coco_eval_bbox')
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                # Only the metric named by `checkpoint_metric` decides best.pth. Without this,
                # whichever key the loop happened to visit last would decide it, which is a
                # silent dependency on dict ordering once a second metric exists.
                if k != selection_metric:
                    continue

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                    print(f"saved best.pth at epoch {epoch} on {selection_metric}"
                          f"={test_stats[k][0]:.4f}")

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
