"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime
import shutil
from pathlib import Path

import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from .early_stopping import EarlyStopping


class DetSolver(BaseSolver):

    def train(self, ):
        self.early_stopping = EarlyStopping(**(self.cfg.yaml_cfg.get('early_stopping') or {}))
        super().train()
        if self.cfg.resume:
            self._carry_over_best()

    def _carry_over_best(self, ):
        """Keep the restored best backed by its weights when resuming into a new output dir
        """
        es = self.early_stopping
        if es.best_epoch < 0:
            return

        best_pth = self.output_dir / 'best.pth'
        resume_dir = Path(self.cfg.resume).parent
        candidates = [best_pth, resume_dir / 'best.pth', resume_dir / f'checkpoint{es.best_epoch:04}.pth']
        for path in candidates:
            if path.exists() and torch.load(path, map_location='cpu').get('last_epoch') == es.best_epoch:
                if path != best_pth and dist_utils.is_main_process():
                    shutil.copyfile(path, best_pth)
                print(f'best.pth: epoch {es.best_epoch} (mAP {es.best_map:.4f}) from {path}')
                return

        print(f'WARNING: weights for the best epoch {es.best_epoch} (mAP {es.best_map:.4f}) not found '
              f'next to {self.cfg.resume}. best.pth will only cover the resumed epochs.')
        es.best_map, es.best_epoch = float('-inf'), -1

    def _write_early_stopping_summary(self, ):
        if not (self.output_dir and dist_utils.is_main_process()):
            return
        es = self.early_stopping
        summary = {
            'enabled': es.enabled,
            'patience': es.patience,
            'min_delta': es.min_delta,
            'epochs_budget': self.cfg.epoches,
            'early_stopped': es.stopped_epoch is not None,
            'stopped_epoch': self.last_epoch,
            'best_epoch': es.best_epoch,
            'best_map': es.best_map if es.best_epoch >= 0 else None,
        }
        with (self.output_dir / 'early_stopping.json').open('w') as f:
            json.dump(summary, f, indent=2)

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg
        es = self.early_stopping

        with open(str(self.output_dir / 'config.txt'), 'w') as f:
            f.write(str(self.cfg.__dict__))

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')
        print(es)

        if es.stopped_epoch is not None:
            if es.enabled:
                print(f'This run already early stopped at epoch {es.stopped_epoch}, nothing to train. '
                      'Set early_stopping.enabled=False to train on.')
                self._write_early_stopping_summary()
                return
            es.stopped_epoch = None

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

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

            is_best = es.step(test_stats['coco_eval_bbox'][0], epoch)
            stop = es.should_stop and epoch < args.epoches - 1
            if stop:
                es.stopped_epoch = epoch

            # Saved after eval, so a resume picks up this epoch's early stopping state
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                if is_best:
                    checkpoint_paths.append(self.output_dir / 'best.pth')
                state = self.state_dict()
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(state, checkpoint_path)

            print(f'best_stat: {dict(epoch=es.best_epoch, coco_eval_bbox=es.best_map)}')
            if es.enabled:
                print(f'early_stopping: {es.wait}/{es.patience} epochs without a {es.min_delta} gain')

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

            if stop:
                print(f'Early stopping at epoch {epoch}: best epoch {es.best_epoch} '
                      f'(mAP {es.best_map:.4f}), no {es.min_delta} gain for {es.patience} epochs')
                break

        self._write_early_stopping_summary()

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
