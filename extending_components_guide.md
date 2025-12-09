# LeapAI框架扩展现有组件指南

## 🎯 概述

本指南将详细介绍如何修改和扩展LeapAI框架中的现有组件。我们将涵盖模型组件、数据处理、损失函数、评估指标等各个方面的扩展方法。

## 📋 扩展场景

### 常见扩展需求
1. **模型架构改进**: 优化backbone、neck、head等
2. **数据处理增强**: 添加新的数据增强策略
3. **损失函数优化**: 改进或添加新的损失函数
4. **评估指标扩展**: 添加新的评估指标
5. **工具函数增强**: 扩展辅助工具和实用函数

## 🚀 扩展实践

### 场景1：改进动态检测头

#### 1.1 扩展现有的DynamicHead
```python
# projects/perception/model/head/enhanced_dynamic_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from mmdet.models import losses

from projects.perception.model.head.dynamic_head_bin import DynamicHead_Bin

class EnhancedDynamicHead(DynamicHead_Bin):
    """增强的动态检测头"""
    
    def __init__(self, 
                 # 继承原有参数
                 bev_h, bev_w, num_query, embed_dims, 
                 bin_cls_num, overlap, occude_cls, code_size,
                 topk_query, queue_length, class_names,
                 enable_temporal, sync_cls_avg_factor,
                 with_box_refine, as_two_stage, decoder,
                 bbox_coder, loss_cls, loss_bin_cls,
                 loss_occlude, loss_attr, loss_bbox,
                 assigner, sampler, loss_weights, cost_weights,
                 only_train_attr, attr_branch_cfg, attr_param,
                 # 新增参数
                 use_focal_loss=True,
                 use_iou_loss=True,
                 use_auxiliary_head=True,
                 attention_type='deformable'):
        super().__init__(
            bev_h, bev_w, num_query, embed_dims, 
            bin_cls_num, overlap, occude_cls, code_size,
            topk_query, queue_length, class_names,
            enable_temporal, sync_cls_avg_factor,
            with_box_refine, as_two_stage, decoder,
            bbox_coder, loss_cls, loss_bin_cls,
            loss_occlude, loss_attr, loss_bbox,
            assigner, sampler, loss_weights, cost_weights,
            only_train_attr, attr_branch_cfg, attr_param
        )
        
        # 新增组件
        self.use_focal_loss = use_focal_loss
        self.use_iou_loss = use_iou_loss
        self.use_auxiliary_head = use_auxiliary_head
        self.attention_type = attention_type
        
        # 改进的注意力机制
        if attention_type == 'multi_scale':
            self.multi_scale_attention = self._build_multi_scale_attention()
        elif attention_type == 'efficient':
            self.efficient_attention = self._build_efficient_attention()
        
        # 辅助检测头
        if use_auxiliary_head:
            self.auxiliary_head = self._build_auxiliary_head()
        
        # IoU感知损失
        if use_iou_loss:
            self.iou_loss = losses.IoULoss()
    
    def _build_multi_scale_attention(self):
        """构建多尺度注意力"""
        return nn.ModuleDict({
            'scale1': nn.MultiheadAttention(self.embed_dims, 8),
            'scale2': nn.MultiheadAttention(self.embed_dims, 8),
            'scale3': nn.MultiheadAttention(self.embed_dims, 8),
        })
    
    def _build_efficient_attention(self):
        """构建高效注意力"""
        return nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
        )
    
    def _build_auxiliary_head(self):
        """构建辅助检测头"""
        return nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dims // 2, self.num_classes),
        )
    
    def forward(self, bev_feats, metas):
        """增强的前向传播"""
        # 调用原始前向传播
        original_output = super().forward(bev_feats, metas)
        
        # 应用增强功能
        if self.use_auxiliary_head:
            # 添加辅助分类损失
            aux_cls_logits = self.auxiliary_head(bev_feats)
            original_output['aux_cls_logits'] = aux_cls_logits
        
        if hasattr(self, 'multi_scale_attention'):
            # 应用多尺度注意力
            enhanced_feats = self._apply_multi_scale_attention(bev_feats)
            original_output['enhanced_feats'] = enhanced_feats
        
        return original_output
    
    def _apply_multi_scale_attention(self, bev_feats):
        """应用多尺度注意力"""
        # 实现多尺度注意力逻辑
        B, C, H, W = bev_feats.shape
        
        # 生成多尺度特征
        scale1 = F.avg_pool2d(bev_feats, 2, 2)
        scale2 = F.avg_pool2d(bev_feats, 4, 4)
        scale3 = bev_feats
        
        # 应用注意力
        scales = [scale1, scale2, scale3]
        attended_scales = []
        
        for i, (scale, attn) in enumerate(zip(scales, self.multi_scale_attention.values())):
            B_s, C_s, H_s, W_s = scale.shape
            scale_flat = scale.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            attended, _ = attn(scale_flat, scale_flat, scale_flat)
            attended = attended.transpose(1, 2).view(B_s, C_s, H_s, W_s)
            attended_scales.append(F.interpolate(attended, size=(H, W), mode='bilinear'))
        
        # 融合多尺度特征
        enhanced_feats = sum(attended_scales) / len(attended_scales)
        return enhanced_feats
    
    def loss(self, gt_boxes, gt_labels, gt_masks, gt_instances, 
             gt_occlude_weight, head_pred, gt_attributes=None, 
             gt_pillar_mask=None):
        """增强的损失计算"""
        # 获取原始损失
        original_losses = super().loss(
            gt_boxes, gt_labels, gt_masks, gt_instances,
            gt_occlude_weight, head_pred, gt_attributes, gt_pillar_mask
        )
        
        # 添加增强损失
        enhanced_losses = {}
        
        if self.use_auxiliary_head and 'aux_cls_logits' in head_pred:
            # 辅助分类损失
            aux_loss = self.loss_cls(
                head_pred['aux_cls_logits'], gt_labels, gt_masks
            )
            enhanced_losses['loss_aux_cls'] = aux_loss * 0.5
        
        if self.use_iou_loss and 'all_cls_scores' in head_pred:
            # IoU感知损失
            iou_loss = self._compute_iou_loss(
                head_pred, gt_boxes, gt_labels, gt_masks
            )
            enhanced_losses['loss_iou'] = iou_loss * 0.3
        
        # 合并损失
        total_losses = {**original_losses, **enhanced_losses}
        total_loss = sum(total_losses.values())
        total_losses['loss'] = total_loss
        
        return total_losses
    
    def _compute_iou_loss(self, head_pred, gt_boxes, gt_labels, gt_masks):
        """计算IoU损失"""
        # 实现IoU损失计算
        # 这里简化实现，实际需要完整的IoU计算
        return torch.tensor(0.0, device=gt_boxes.device)
```

#### 1.2 在配置中使用增强的检测头
```python
# 在动态任务配置中使用增强的检测头
# projects/perception/dynamic_enhanced.py

from projects.perception.model.head.enhanced_dynamic_head import EnhancedDynamicHead

# 替换原有的检测头
enhanced_dynamic_head = dict(
    type=EnhancedDynamicHead,
    bev_h=bev_h,
    bev_w=bev_w,
    num_query=384,
    embed_dims=embed_dims,
    bin_cls_num=bins,
    overlap=overlap,
    occude_cls=1,
    code_size=6 + bins + 3,
    topk_query=int(384 / 3) if enable_temporal else 0,
    queue_length=4 if enable_temporal else 0,
    class_names=class_names,
    enable_temporal=enable_temporal,
    sync_cls_avg_factor=True,
    with_box_refine=True,
    as_two_stage=False,
    # 新增参数
    use_focal_loss=True,
    use_iou_loss=True,
    use_auxiliary_head=True,
    attention_type='multi_scale',
)

# 更新节点配置
nodes["bev_dynamic_enhanced_head"] = enhanced_dynamic_head
```

### 场景2：扩展数据增强

#### 2.1 创建新的数据增强策略
```python
# projects/perception/transforms/advanced_augmentation.py
import random
import numpy as np
import torch
import cv2
from torchvision import transforms

class AdvancedAugmentation:
    """高级数据增强策略"""
    
    def __init__(self, 
                 weather_augmentation=True,
                 lighting_augmentation=True,
                 motion_blur=True,
                 noise_injection=True,
                 cutmix_prob=0.5,
                 mixup_prob=0.3):
        self.weather_augmentation = weather_augmentation
        self.lighting_augmentation = lighting_augmentation
        self.motion_blur = motion_blur
        self.noise_injection = noise_injection
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
    
    def __call__(self, images, metas=None):
        """应用高级增强"""
        if random.random() < 0.3:  # 30%概率应用增强
            images = self._apply_weather_augmentation(images)
        
        if random.random() < 0.4:  # 40%概率应用光照增强
            images = self._apply_lighting_augmentation(images)
        
        if random.random() < 0.2:  # 20%概率应用运动模糊
            images = self._apply_motion_blur(images)
        
        if random.random() < 0.3:  # 30%概率注入噪声
            images = self._apply_noise_injection(images)
        
        return images, metas
    
    def _apply_weather_augmentation(self, images):
        """应用天气增强"""
        # 模拟雨天效果
        if random.random() < 0.3:
            images = self._simulate_rain(images)
        
        # 模拟雾天效果
        elif random.random() < 0.3:
            images = self._simulate_fog(images)
        
        # 模拟雪天效果
        elif random.random() < 0.3:
            images = self._simulate_snow(images)
        
        return images
    
    def _simulate_rain(self, images):
        """模拟雨天效果"""
        rain_images = []
        for img in images:
            # 添加雨线效果
            h, w = img.shape[-2:]
            rain_mask = np.random.random((h, w)) > 0.95
            rain_lines = np.random.random((h, w)) * 0.1
            
            # 应用雨线
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np[rain_mask] += rain_lines[rain_mask]
            img_np = np.clip(img_np, 0, 1)
            
            rain_img = torch.from_numpy(img_np).permute(2, 0, 1).to(img.device)
            rain_images.append(rain_img)
        
        return rain_images
    
    def _simulate_fog(self, images):
        """模拟雾天效果"""
        fog_images = []
        for img in images:
            # 添加雾效果
            fog_intensity = random.uniform(0.1, 0.3)
            fog_mask = np.ones_like(img.permute(1, 2, 0).cpu().numpy()) * (1 - fog_intensity)
            
            img_np = img.permute(1, 2, 0).cpu().numpy()
            foggy_img = img_np * fog_mask + fog_intensity
            
            fog_img = torch.from_numpy(foggy_img).permute(2, 0, 1).to(img.device)
            fog_images.append(fog_img)
        
        return fog_images
    
    def _apply_lighting_augmentation(self, images):
        """应用光照增强"""
        # 随机调整亮度、对比度、饱和度
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        
        enhanced_images = []
        for img in images:
            # 应用颜色变换
            enhancer = transforms.ColorJitter(
                brightness=brightness_factor - 1.0,
                contrast=contrast_factor - 1.0,
                saturation=saturation_factor - 1.0,
                hue=0
            )
            enhanced_img = enhancer(img)
            enhanced_images.append(enhanced_img)
        
        return enhanced_images
    
    def _apply_motion_blur(self, images):
        """应用运动模糊"""
        blurred_images = []
        for img in images:
            # 随机运动模糊核
            kernel_size = random.choice([3, 5, 7])
            angle = random.uniform(0, 360)
            
            # 创建运动模糊核
            kernel = self._create_motion_blur_kernel(kernel_size, angle)
            
            # 应用卷积
            img_np = img.permute(1, 2, 0).cpu().numpy()
            blurred_img = cv2.filter2D(img_np, kernel, cv2.BORDER_REFLECT)
            
            blurred_tensor = torch.from_numpy(blurred_img).permute(2, 0, 1).to(img.device)
            blurred_images.append(blurred_tensor)
        
        return blurred_images
    
    def _create_motion_blur_kernel(self, kernel_size, angle):
        """创建运动模糊核"""
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # 计算运动方向
        angle_rad = np.radians(angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # 生成运动线
        for i in range(kernel_size):
            for j in range(kernel_size):
                pos = np.array([i - center, j - center])
                # 计算到运动线的距离
                distance = abs(np.cross(direction, pos))
                if distance < 1:
                    kernel[i, j] = 1
        
        # 归一化
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        return kernel.astype(np.float32)
    
    def _apply_noise_injection(self, images):
        """注入噪声"""
        noisy_images = []
        for img in images:
            # 高斯噪声
            noise = torch.randn_like(img) * 0.02
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
            noisy_images.append(noisy_img)
        
        return noisy_images
```

#### 2.2 在数据管道中使用高级增强
```python
# 在训练管道中添加高级增强
train_pipeline = [
    # ... 其他变换
    dict(
        type=AdvancedAugmentation,
        weather_augmentation=True,
        lighting_augmentation=True,
        motion_blur=True,
        noise_injection=True,
        cutmix_prob=0.5,
        mixup_prob=0.3,
    ),
    # ... 其他变换
]
```

### 场景3：扩展损失函数

#### 3.1 创建新的损失函数
```python
# leapai/model/loss/advanced_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import losses

class FocalTverskyLoss(nn.Module):
    """Focal Tversky损失"""
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        # 计算Tversky指数
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        
        tversky = (intersection + 1e-6) / (intersection + self.alpha * fp + self.beta * fn + 1e-6)
        
        # 应用Focal权重
        focal_weight = (1 - tversky) ** self.gamma
        loss = focal_weight * tversky
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AdaptiveBoxLoss(nn.Module):
    """自适应边界框损失"""
    
    def __init__(self, beta=1.0, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps
    
    def forward(self, pred_boxes, gt_boxes, weights=None):
        """
        Args:
            pred_boxes: [N, 7] (x, y, z, w, l, h, theta)
            gt_boxes: [N, 7]
            weights: [N] optional weights
        """
        # 分解边界框
        pred_center = pred_boxes[:, :3]  # x, y, z
        pred_size = pred_boxes[:, 3:6]  # w, l, h
        pred_angle = pred_boxes[:, 6]   # theta
        
        gt_center = gt_boxes[:, :3]
        gt_size = gt_boxes[:, 3:6]
        gt_angle = gt_boxes[:, 6]
        
        # 中心点损失
        center_loss = F.l1_loss(pred_center, gt_center, reduction='none')
        
        # 尺寸损失（自适应权重）
        size_diff = torch.abs(pred_size - gt_size)
        size_weights = torch.exp(-self.beta * size_diff.mean(dim=1, keepdim=True))
        size_loss = (size_diff * size_weights).mean()
        
        # 角度损失
        angle_diff = torch.abs(pred_angle - gt_angle)
        # 处理角度周期性
        angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
        angle_loss = (angle_diff ** 2).mean()
        
        # 组合损失
        total_loss = center_loss.mean() + size_loss + angle_loss
        
        # 应用权重
        if weights is not None:
            total_loss = total_loss * weights.unsqueeze(1)
        
        return total_loss.mean()

class MultiTaskLoss(nn.Module):
    """多任务损失平衡"""
    
    def __init__(self, task_names, loss_weights=None, adaptive_weights=True):
        super().__init__()
        self.task_names = task_names
        self.adaptive_weights = adaptive_weights
        
        if loss_weights is None:
            self.loss_weights = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(1.0)) 
                for task in task_names
            })
        else:
            self.loss_weights = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(weight)) 
                for task, weight in loss_weights.items()
            })
    
    def forward(self, losses_dict):
        """
        Args:
            losses_dict: dict of task losses
        """
        total_loss = 0
        weighted_losses = {}
        
        for task_name in self.task_names:
            if task_name in losses_dict:
                task_loss = losses_dict[task_name]
                task_weight = self.loss_weights[task_name]
                
                # 自适应权重调整
                if self.adaptive_weights:
                    # 基于损失大小动态调整权重
                    loss_magnitude = task_loss.item() if torch.is_tensor(task_loss) else task_loss
                    adaptive_weight = task_weight / (1.0 + loss_magnitude)
                    weighted_loss = task_loss * adaptive_weight
                else:
                    weighted_loss = task_loss * task_weight
                
                total_loss += weighted_loss
                weighted_losses[f'weighted_{task_name}'] = weighted_loss
        
        weighted_losses['total_loss'] = total_loss
        return weighted_losses
```

#### 3.2 在模型中使用新的损失函数
```python
# 在检测头中使用新的损失函数
class EnhancedDynamicHead(DynamicHead_Bin):
    def __init__(self, ...):
        # ... 原有初始化
        
        # 新增损失函数
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2.0)
        self.adaptive_box_loss = AdaptiveBoxLoss(beta=1.0)
        self.multi_task_loss = MultiTaskLoss(
            task_names=['cls', 'bbox', 'aux'],
            adaptive_weights=True
        )
    
    def loss(self, gt_boxes, gt_labels, gt_masks, ...):
        # ... 原有损失计算
        
        # 应用新的损失函数
        losses = {}
        
        # Focal Tversky损失用于分类
        if 'all_cls_scores' in head_pred:
            focal_tversky_loss = self.focal_tversky_loss(
                head_pred['all_cls_scores'], 
                gt_labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            )
            losses['loss_focal_tversky'] = focal_tversky_loss
        
        # 自适应边界框损失
        if 'all_bbox_preds' in head_pred:
            adaptive_box_loss = self.adaptive_box_loss(
                head_pred['all_bbox_preds'], gt_boxes
            )
            losses['loss_adaptive_bbox'] = adaptive_box_loss
        
        # 多任务损失平衡
        multi_task_losses = self.multi_task_loss(losses)
        losses.update(multi_task_losses)
        
        return losses
```

### 场景4：扩展评估指标

#### 4.1 创建新的评估指标
```python
# projects/perception/callback/metric/enhanced_metrics.py
import numpy as np
from leapai.callback.metric.base_metric import BaseMetric

class EnhancedDetectionMetric(BaseMetric):
    """增强的检测评估指标"""
    
    def __init__(self, 
                 task_name,
                 annotation_name,
                 save_dir,
                 class_names,
                 distance_thresholds=[0.5, 1.0, 2.0, 4.0],
                 score_thresholds=np.arange(0.1, 1.0, 0.1),
                 evaluate_speed=True,
                 evaluate_size_accuracy=True):
        super().__init__()
        self.task_name = task_name
        self.annotation_name = annotation_name
        self.save_dir = save_dir
        self.class_names = class_names
        self.distance_thresholds = distance_thresholds
        self.score_thresholds = score_thresholds
        self.evaluate_speed = evaluate_speed
        self.evaluate_size_accuracy = evaluate_size_accuracy
        
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.predictions = []
        self.ground_truths = []
        self.inference_times = []
    
    def process(self, predictions, ground_truth, inference_time=None):
        """处理单个样本的预测和真值"""
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truth)
        
        if inference_time is not None:
            self.inference_times.append(inference_time)
    
    def compute_metrics(self):
        """计算增强的评估指标"""
        results = {}
        
        # 1. 多阈值mAP
        ap_results = {}
        for threshold in self.distance_thresholds:
            ap_per_class = []
            for class_id, class_name in enumerate(self.class_names):
                ap = self._compute_ap_at_threshold(threshold, class_id)
                ap_per_class.append(ap)
                print(f"{class_name} AP@{threshold}m: {ap:.4f}")
            
            ap_results[f'mAP@{threshold}m'] = np.mean(ap_per_class)
            ap_results[f'AP_per_class@{threshold}m'] = dict(zip(self.class_names, ap_per_class))
        
        results.update(ap_results)
        
        # 2. 多分数阈值评估
        score_results = {}
        for score_thresh in self.score_thresholds:
            precision, recall = self._compute_precision_recall_at_score(score_thresh)
            score_results[f'precision@{score_thresh:.1f}'] = precision
            score_results[f'recall@{score_thresh:.1f}'] = recall
        
        results.update(score_results)
        
        # 3. 推理速度评估
        if self.evaluate_speed and self.inference_times:
            avg_inference_time = np.mean(self.inference_times)
            fps = 1.0 / avg_inference_time
            results['avg_inference_time'] = avg_inference_time
            results['fps'] = fps
            print(f"平均推理时间: {avg_inference_time:.4f}s, FPS: {fps:.2f}")
        
        # 4. 尺寸精度评估
        if self.evaluate_size_accuracy:
            size_accuracy = self._compute_size_accuracy()
            results['size_accuracy'] = size_accuracy
        
        return results
    
    def _compute_ap_at_threshold(self, distance_threshold, class_id):
        """计算特定距离阈值下的AP"""
        # 实现AP计算逻辑
        # 这里简化实现
        return 0.0  # 占位符
    
    def _compute_precision_recall_at_score(self, score_threshold):
        """计算特定分数阈值下的精确率和召回率"""
        # 实现精确率和召回率计算
        return 0.0, 0.0  # 占位符
    
    def _compute_size_accuracy(self):
        """计算尺寸估计精度"""
        # 实现尺寸精度计算
        return 0.0  # 占位符
    
    def save_results(self, results):
        """保存增强的评估结果"""
        import json
        import os
        
        save_path = os.path.join(self.save_dir, f"{self.task_name}_enhanced_results.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"增强评估结果已保存到: {save_path}")
        
        # 生成详细的评估报告
        self._generate_evaluation_report(results)
    
    def _generate_evaluation_report(self, results):
        """生成详细的评估报告"""
        report_path = os.path.join(self.save_dir, f"{self.task_name}_evaluation_report.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# {self.task_name} 评估报告\n\n")
            
            # 多阈值mAP结果
            f.write("## 多阈值mAP结果\n\n")
            f.write("| 距离阈值 | mAP |\n")
            f.write("|------------|-----|\n")
            for threshold in self.distance_thresholds:
                map_key = f'mAP@{threshold}m'
                if map_key in results:
                    f.write(f"| {threshold}m | {results[map_key]:.4f} |\n")
            
            # 类别详细结果
            f.write("\n## 各类别AP结果\n\n")
            for class_name in self.class_names:
                f.write(f"### {class_name}\n")
                for threshold in self.distance_thresholds:
                    ap_key = f'AP_per_class@{threshold}m'
                    if ap_key in results and class_name in results[ap_key]:
                        f.write(f"- AP@{threshold}m: {results[ap_key][class_name]:.4f}\n")
                f.write("\n")
        
        print(f"评估报告已生成: {report_path}")
```

### 场景5：扩展工具函数

#### 5.1 创建实用工具
```python
# projects/perception/utils/enhanced_utils.py
import torch
import numpy as np
import cv2
from typing import List, Dict, Any

class EnhancedVisualization:
    """增强的可视化工具"""
    
    def __init__(self, save_dir="./visualization"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_detection_results(self, images, predictions, ground_truth, save_name="detection_vis"):
        """可视化检测结果"""
        import matplotlib.pyplot as plt
        
        batch_size = len(images)
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
        
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            pred_boxes = predictions[i].get('boxes_3d', [])
            gt_boxes = ground_truth[i].get('boxes_3d', [])
            
            # 原图
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Original Image {i}")
            axes[i, 0].axis('off')
            
            # 预测结果
            pred_img = self._draw_boxes_on_image(img.copy(), pred_boxes, color='red', label='pred')
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f"Predictions {i}")
            axes[i, 1].axis('off')
            
            # 真值结果
            gt_img = self._draw_boxes_on_image(img.copy(), gt_boxes, color='green', label='gt')
            axes[i, 2].imshow(gt_img)
            axes[i, 2].set_title(f"Ground Truth {i}")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _draw_boxes_on_image(self, img, boxes, color='red', label='pred'):
        """在图像上绘制边界框"""
        img_with_boxes = img.copy()
        
        for box in boxes:
            if len(box) >= 7:  # x, y, z, w, l, h, theta
                x, y, z, w, l, h, theta = box[:7]
                
                # 简化：只在2D图像上绘制中心点
                center_x = int(x * img.shape[1] / 100)  # 假设BEV坐标范围
                center_y = int(y * img.shape[0] / 100)
                
                # 绘制边界框
                cv2.rectangle(img_with_boxes, 
                           (center_x-10, center_y-10), 
                           (center_x+10, center_y+10), 
                           color, 2)
                
                # 添加标签
                cv2.putText(img_with_boxes, label, 
                           (center_x-15, center_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img_with_boxes

class ModelProfiler:
    """模型性能分析工具"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置分析器"""
        self.layer_times = {}
        self.memory_usage = []
        self.flops_count = 0
    
    def profile_layer(self, layer_name, input_tensor, output_tensor):
        """分析特定层的性能"""
        import time
        
        start_time = time.time()
        
        # 模拟层计算
        with torch.no_grad():
            _ = output_tensor  # 确保输出被计算
        
        end_time = time.time()
        
        layer_time = end_time - start_time
        if layer_name not in self.layer_times:
            self.layer_times[layer_name] = []
        self.layer_times[layer_name].append(layer_time)
        
        # 记录内存使用
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_usage.append(memory_used)
    
    def get_profile_report(self):
        """获取性能分析报告"""
        report = {}
        
        # 层时间分析
        for layer_name, times in self.layer_times.items():
            report[f'{layer_name}_avg_time'] = np.mean(times)
            report[f'{layer_name}_total_time'] = np.sum(times)
            report[f'{layer_name}_call_count'] = len(times)
        
        # 内存使用分析
        if self.memory_usage:
            report['avg_memory_usage'] = np.mean(self.memory_usage)
            report['peak_memory_usage'] = np.max(self.memory_usage)
        
        return report

class DataAnalyzer:
    """数据分析工具"""
    
    @staticmethod
    def analyze_dataset_statistics(dataset_path):
        """分析数据集统计信息"""
        import json
        import os
        
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # 分析统计信息
            stats = {
                'total_samples': len(data) if isinstance(data, list) else 1,
                'sample_keys': list(data[0].keys()) if data else [],
                'file_size': os.path.getsize(dataset_path) / 1024**2,  # MB
            }
            
            # 如果是列表，分析每个样本
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                if 'annotations' in sample:
                    annotations = sample['annotations']
                    stats.update({
                        'avg_annotations_per_sample': len(annotations),
                        'annotation_types': list(set(ann.get('type', 'unknown') for ann in annotations))
                    })
            
            return stats
        else:
            return {'error': f'Dataset file not found: {dataset_path}'}
    
    @staticmethod
    def visualize_class_distribution(labels, class_names, save_path=None):
        """可视化类别分布"""
        import matplotlib.pyplot as plt
        
        # 统计每个类别的数量
        class_counts = {}
        for label in labels:
            class_name = class_names[label] if label < len(class_names) else 'unknown'
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 绘制分布图
        plt.figure(figsize=(12, 6))
        
        # 柱状图
        plt.subplot(1, 2, 1)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(classes, counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 饼图
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, autopct='%1.1f%%')
        plt.title('Class Distribution (Pie)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return class_counts
```

## 🧪 测试和验证

### 测试扩展组件
```python
def test_enhanced_components():
    """测试增强组件"""
    print("=== 测试增强组件 ===")
    
    # 1. 测试增强的检测头
    test_enhanced_head()
    
    # 2. 测试高级数据增强
    test_advanced_augmentation()
    
    # 3. 测试新的损失函数
    test_enhanced_losses()
    
    # 4. 测试增强的评估指标
    test_enhanced_metrics()
    
    print("所有增强组件测试完成!")

def test_enhanced_head():
    """测试增强的检测头"""
    from projects.perception.model.head.enhanced_dynamic_head import EnhancedDynamicHead
    
    # 创建测试数据
    batch_size, embed_dims, bev_h, bev_w = 2, 256, 128, 128
    bev_features = torch.randn(batch_size, embed_dims, bev_h, bev_w)
    
    # 创建增强检测头
    head = EnhancedDynamicHead(
        bev_h=bev_h, bev_w=bev_w, num_query=384, embed_dims=embed_dims,
        bin_cls_num=8, overlap=0.1, occude_cls=1, code_size=17,
        topk_query=128, queue_length=4, class_names=['car', 'truck'],
        enable_temporal=False, sync_cls_avg_factor=True,
        with_box_refine=True, as_two_stage=False,
        use_focal_loss=True, use_iou_loss=True, use_auxiliary_head=True
    )
    
    # 前向传播
    with torch.no_grad():
        predictions = head(bev_features)
    
    print(f"增强检测头测试成功!")
    print(f"输出键: {list(predictions.keys())}")
    return True

def test_advanced_augmentation():
    """测试高级数据增强"""
    from projects.perception.transforms.advanced_augmentation import AdvancedAugmentation
    
    # 创建测试图像
    batch_size, channels, height, width = 2, 3, 224, 224
    images = torch.rand(batch_size, channels, height, width)
    
    # 创建增强器
    augmenter = AdvancedAugmentation(
        weather_augmentation=True,
        lighting_augmentation=True,
        motion_blur=True,
        noise_injection=True
    )
    
    # 应用增强
    augmented_images, _ = augmenter(images)
    
    print(f"高级数据增强测试成功!")
    print(f"原始图像形状: {images.shape}")
    print(f"增强图像形状: {augmented_images.shape}")
    return True

def test_enhanced_losses():
    """测试增强的损失函数"""
    from leapai.model.loss.advanced_losses import FocalTverskyLoss, AdaptiveBoxLoss
    
    # 创建测试数据
    batch_size, num_classes = 4, 5
    pred = torch.randn(batch_size, num_classes, 64, 64)
    target = torch.randint(0, 2, (batch_size, 64, 64)).float()
    
    # 测试Focal Tversky损失
    focal_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2.0)
    loss1 = focal_loss(pred, target)
    
    # 测试自适应边界框损失
    pred_boxes = torch.randn(batch_size, 7)
    gt_boxes = torch.randn(batch_size, 7)
    adaptive_loss = AdaptiveBoxLoss(beta=1.0)
    loss2 = adaptive_loss(pred_boxes, gt_boxes)
    
    print(f"增强损失函数测试成功!")
    print(f"Focal Tversky损失: {loss1.item():.4f}")
    print(f"自适应边界框损失: {loss2.item():.4f}")
    return True

def test_enhanced_metrics():
    """测试增强的评估指标"""
    from projects.perception.callback.metric.enhanced_metrics import EnhancedDetectionMetric
    
    # 创建测试数据
    predictions = [
        {'boxes_3d': np.random.rand(5, 7), 'scores': np.random.rand(5), 'labels': np.random.randint(0, 3, 5)}
        for _ in range(10)
    ]
    
    ground_truth = [
        {'boxes_3d': np.random.rand(3, 7), 'labels': np.random.randint(0, 3, 3)}
        for _ in range(10)
    ]
    
    # 创建评估器
    metric = EnhancedDetectionMetric(
        task_name="test",
        annotation_name="test_annos",
        save_dir="./test_results",
        class_names=["car", "truck", "person"],
        distance_thresholds=[0.5, 1.0, 2.0],
        evaluate_speed=True
    )
    
    # 处理数据
    for pred, gt in zip(predictions, ground_truth):
        metric.process([pred], [gt], inference_time=0.1)
    
    # 计算指标
    results = metric.compute_metrics()
    
    print(f"增强评估指标测试成功!")
    print(f"评估结果键: {list(results.keys())}")
    return True

# 运行测试
if __name__ == "__main__":
    test_enhanced_components()
```

## 🚀 部署和使用

### 在配置中集成扩展组件
```python
# 创建扩展的配置文件
# projects/perception/configs/enhanced_perception.py

# 导入基础配置
from .lpperception_current_hpa_step1 import *

# 启用增强功能
use_enhanced_head = True
use_advanced_augmentation = True
use_enhanced_losses = True
use_enhanced_metrics = True

# 更新模型配置
if use_enhanced_head:
    from projects.perception.model.head.enhanced_dynamic_head import EnhancedDynamicHead
    
    # 替换检测头
    enhanced_head = dict(
        type=EnhancedDynamicHead,
        # ... 参数配置
    )
    nodes["bev_dynamic_head"] = enhanced_head

# 更新数据管道
if use_advanced_augmentation:
    from projects.perception.transforms.advanced_augmentation import AdvancedAugmentation
    
    # 添加高级增强
    advanced_aug = dict(
        type=AdvancedAugmentation,
        weather_augmentation=True,
        lighting_augmentation=True,
        motion_blur=True,
        noise_injection=True,
    )
    train_pipeline.insert(-2, advanced_aug)  # 在目标处理前添加

# 更新评估指标
if use_enhanced_metrics:
    from projects.perception.callback.metric.enhanced_metrics import EnhancedDetectionMetric
    
    # 使用增强的评估指标
    def get_enhanced_metric(test_set_name):
        return dict(
            type=EnhancedDetectionMetric,
            task_name=test_set_name,
            annotation_name=anno_name,
            save_dir=save_root,
            class_names=class_names,
            distance_thresholds=[0.5, 1.0, 2.0, 4.0],
            evaluate_speed=True,
            evaluate_size_accuracy=True,
        )
    
    # 替换原有的metric函数
    get_metric = get_enhanced_metric
```

## 📊 性能优化建议

### 1. 模型优化
- **注意力机制优化**: 使用高效注意力变体
- **特征融合优化**: 改进多尺度特征融合
- **损失函数优化**: 自适应权重调整

### 2. 数据处理优化
- **增强策略优化**: 智能增强选择
- **数据加载优化**: 并行处理和缓存
- **内存管理**: 动态内存分配

### 3. 训练优化
- **学习率调度**: 自适应学习率调整
- **梯度优化**: 梯度裁剪和累积
- **正则化**: 防止过拟合

## 🔧 常见问题解决

### 1. 兼容性问题
```python
def check_component_compatibility():
    """检查组件兼容性"""
    # 检查版本兼容性
    # 检查接口兼容性
    # 检查数据格式兼容性
    pass

check_component_compatibility()
```

### 2. 性能问题
```python
def debug_performance_issues():
    """调试性能问题"""
    # 使用性能分析器
    # 检查内存泄漏
    # 优化计算瓶颈
    pass

debug_performance_issues()
```

### 3. 集成问题
```python
def solve_integration_issues():
    """解决集成问题"""
    # 检查配置冲突
    # 验证数据流
    # 测试端到端流程
    pass

solve_integration_issues()
```

## 🎯 总结

通过本指南，您已经学会了：

1. **模型扩展**: 如何改进现有的模型组件
2. **数据增强**: 如何添加高级数据增强策略
3. **损失函数**: 如何实现新的损失函数
4. **评估指标**: 如何扩展评估指标
5. **工具函数**: 如何创建实用的辅助工具
6. **测试验证**: 如何全面测试扩展组件
7. **部署集成**: 如何将扩展组件集成到框架中

## 📚 扩展建议

1. **持续优化**: 基于实际使用反馈持续改进
2. **模块化设计**: 保持组件的模块化和可复用性
3. **文档完善**: 为扩展组件编写详细文档
4. **性能监控**: 建立性能监控和反馈机制
5. **社区贡献**: 将有用的扩展贡献给社区

---

**注意**: 本指南提供了扩展LeapAI框架组件的完整流程。在实际扩展中，请确保新组件与现有框架的兼容性，并进行充分的测试验证。
