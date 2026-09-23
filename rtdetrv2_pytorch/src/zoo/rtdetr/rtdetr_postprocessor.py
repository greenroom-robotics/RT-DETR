"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # Classes are 1-indexed (hence the num_classes + 1). The 0th index always has a 
        # near-zero score for a trained model
        if not self.training:

            # Ignore 0-th index (see above comment)
            scores = F.sigmoid(logits[:,:,1:])

            ## remove duplicate indices
            scores, index = torch.topk(scores.max(-1).values, self.num_top_queries, dim=-1)

            # Probability of each class
            soft_labels = F.softmax(logits[:,:,1:], dim=-1)
            labels = soft_labels.gather(dim=1, index=index.unsqueeze(-1).tile(1, 1, soft_labels.shape[-1]))
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

            # Existence from the dedicated objectness head
            objectness = F.sigmoid(outputs['pred_obj']).squeeze(-1).gather(dim=1, index=index)

        else:
            if self.use_focal_loss:
                # det_engine.evaluate never calls postprocessor.eval(), so in-training COCO
                # validation runs this branch. Select rows exactly as the deploy branch above
                # does (one row per query, ranked by the class max) so val mAP measures the
                # graph that ships. Only the label shape differs: COCO needs an int id here,
                # the ONNX carries the soft distribution. Class 0 is the dead slot, hence the
                # slice and the +1.
                class_scores = F.sigmoid(logits[:, :, 1:])
                per_query, labels = class_scores.max(-1)
                scores, index = torch.topk(per_query, self.num_top_queries, dim=-1)
                labels = labels.gather(dim=1, index=index) + 1
                boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
                objectness = F.sigmoid(outputs['pred_obj']).squeeze(-1).gather(dim=1, index=index)
                
            else:
                scores = F.softmax(logits)[:, :, :-1]
                scores, labels = scores.max(dim=-1)
                objectness = F.sigmoid(outputs['pred_obj']).squeeze(-1)
                if scores.shape[1] > self.num_top_queries:
                    scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                    labels = torch.gather(labels, dim=1, index=index)
                    boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                    objectness = torch.gather(objectness, dim=1, index=index)
    
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores, objectness

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco, obj in zip(labels, boxes, scores, objectness):
            result = dict(labels=lab, boxes=box, scores=sco, objectness=obj)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
