import torch
import math

def inner_iou(box1, box2, ratio=0.7, eps=1e-7):
    """
    计算Inner-IoU
    Args:
        box1 (torch.Tensor): [..., 4] (x1, y1, x2, y2)
        box2 (torch.Tensor): [..., 4] (x1, y1, x2, y2)
        ratio (float): 内部区域比例 (0-1)
        eps (float): 避免除零的小值
    Returns:
        torch.Tensor: Inner-IoU值
    """
    # 计算原始框的面积
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # 计算原始IoU的交集
    lt = torch.max(box1[..., :2], box2[..., :2])
    rb = torch.min(box1[..., 2:], box2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter + eps
    iou = inter / union
    
    # 计算内部框
    def get_inner_box(box, ratio):
        center_x = (box[..., 0] + box[..., 2]) / 2
        center_y = (box[..., 1] + box[..., 3]) / 2
        width = (box[..., 2] - box[..., 0]) * ratio
        height = (box[..., 3] - box[..., 1]) * ratio
        return torch.stack([
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2
        ], dim=-1)
    
    inner_box1 = get_inner_box(box1, ratio)
    inner_box2 = get_inner_box(box2, ratio)
    
    # 计算内部框的交集
    inner_lt = torch.max(inner_box1[..., :2], inner_box2[..., :2])
    inner_rb = torch.min(inner_box1[..., 2:], inner_box2[..., 2:])
    inner_wh = (inner_rb - inner_lt).clamp(min=0)
    inner_inter = inner_wh[..., 0] * inner_wh[..., 1]
    
    # 计算内部框的最小面积
    inner_area1 = (inner_box1[..., 2] - inner_box1[..., 0]) * (inner_box1[..., 3] - inner_box1[..., 1])
    inner_area2 = (inner_box2[..., 2] - inner_box2[..., 0]) * (inner_box2[..., 3] - inner_box2[..., 1])
    inner_min_area = torch.min(inner_area1, inner_area2)
    
    # 计算Inner-IoU
    inner_iou = inner_inter / (inner_min_area + eps)
    
    # 结合原始IoU和Inner-IoU
    return (iou + inner_iou) / 2
class InnerIoULoss:
    """Inner-IoU损失函数"""
    
    def __init__(self, iou_ratio=0.7, box_format='xywh', reduction='mean'):
        self.iou_ratio = iou_ratio
        self.box_format = box_format
        self.reduction = reduction
        
    def __call__(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测框 [..., 4] (x, y, w, h) 或 (x1, y1, x2, y2)
            target (torch.Tensor): 目标框 [..., 4] (x, y, w, h) 或 (x1, y1, x2, y2)
        Returns:
            torch.Tensor: 损失值
        """
        if self.box_format == 'xywh':
            # 转换xywh为xyxy
            pred_xyxy = torch.cat([
                pred[..., :2] - pred[..., 2:] / 2,
                pred[..., :2] + pred[..., 2:] / 2
            ], dim=-1)
            target_xyxy = torch.cat([
                target[..., :2] - target[..., 2:] / 2,
                target[..., :2] + target[..., 2:] / 2
            ], dim=-1)
        else:
            pred_xyxy = pred
            target_xyxy = target
        
        # 计算Inner-IoU
        iou = inner_iou(pred_xyxy, target_xyxy, ratio=self.iou_ratio)
        
        # 计算损失
        loss = 1 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
from ultralytics.utils.loss import v8DetectionLoss

class v8DetectionLossWithInnerIoU(v8DetectionLoss):
    """使用Inner-IoU的YOLOv8检测损失"""
    
    def __init__(self, model):
        super().__init__(model)
        # 替换box_loss为Inner-IoU损失
        self.box_loss = InnerIoULoss(iou_ratio=0.7, box_format='xywh')
        
    def __call__(self, preds, batch):
        # 保持其他损失不变，只修改框回归损失
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        
        # 前向计算
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        # 计算其他部分损失
        loss_items = self._calculate_loss(pred_distri, pred_scores, batch)
        
        # 使用Inner-IoU计算框回归损失
        bbox_loss = self.box_loss(pred_distri, batch['bboxes'])
        loss[0] = bbox_loss
        
        # 分类损失和DFL损失保持不变
        loss[1] = loss_items[1]  # cls_loss
        loss[2] = loss_items[2]  # dfl_loss
        
        # 总损失
        loss[3] = loss_items[3]  # 保持与原损失计算一致
        
        return loss.sum() * batch['img'].shape[0], loss.detach()