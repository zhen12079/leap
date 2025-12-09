# LeapAI框架添加新感知任务指南

## 🎯 概述

本指南将详细介绍如何在LeapAI框架中添加一个新的感知任务。我们将以添加一个"交通标志检测"任务为例，展示完整的扩展流程。

## 📋 任务分析

### 新任务：交通标志检测
- **任务类型**: 目标检测
- **输入**: BEV特征图
- **输出**: 交通标志的3D边界框和类别
- **类别**: 禁止标志、警告标志、指示标志等

## 🚀 实现步骤

### 步骤1：创建任务配置文件

#### 1.1 创建配置文件
```python
# projects/perception/configs/traffic_sign.py
import os
from copy import deepcopy
from functools import partial

import numpy as np
import torch.nn as nn
from mmdet.models import losses

from projects.perception import base, model_base
from projects.perception.callback.metric.traffic_sign_metric import (
    TrafficSignMetric,
)
from projects.perception.dataset.frame_sampler import FrameSamplerTrafficSign
from projects.perception.dataset.gt_mask import GTMask
from projects.perception.model.head.traffic_sign_head import TrafficSignHead
from projects.perception.target.traffic_sign_target import TrafficSignTarget
from projects.perception.transforms.augment_transfer import GenerateAugMatrix

# 基础配置
save_root = base.save_root
embed_dims = model_base.embed_dims
camera_names = model_base.camera_names
resize_hw = model_base.resize_hw
batch_size = model_base.batch_sizes.get("traffic_sign", 16)
num_workers = model_base.num_workers.get("traffic_sign", 4)
lidar_range = model_base.lidar_range.get("traffic_sign", [-50, -50, -3, 50, 50, 5])

# 任务特定配置
task_name = "traffic_sign"
anno_name = "annos_traffic_sign"
bev_h, bev_w = 128, 128  # BEV特征图尺寸

# 类别定义
class_names = [
    "prohibitory",      # 禁止标志
    "warning",         # 警告标志  
    "mandatory",       # 指示标志
    "priority",        # 优先权标志
    "information",    # 信息标志
]
num_classes = len(class_names)
category2id = {
    "prohibitory": 0,
    "warning": 1,
    "mandatory": 2,
    "priority": 3,
    "information": 4,
}

# 检测配置
max_objects = 100
post_center_range = [-50, -50, -5, 50, 50, 5]
```

#### 1.2 数据处理配置
```python
# 数据增强配置
augment_2d_flag = True
augment_3d_flag = True

if augment_2d_flag:
    data_config_2d = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "resize": (-0.1, 0.1),
        "crop": (-0.05, 0.05),
        "rot": (-3.0, 3.0),
        "flip": True,
    }

if augment_3d_flag:
    data_config_3d = {
        "rotate_z": [1.0, (-10, 10)],
        "scale": (0.9, 1.1),
        "x_trans": (-2, 2),
        "y_trans": (-2, 2),
        "z_trans": (-1, 1),
    }

# 训练数据管道
train_pipeline = []
if augment_2d_flag or augment_3d_flag:
    train_pipeline.append(
        dict(
            type=GenerateAugMatrix,
            apply_names=camera_names,
            resize_hw=resize_hw,
            prob=0.5,
            data_aug_config_2d=data_config_2d if augment_2d_flag else None,
            data_aug_config_3d=data_config_3d if augment_3d_flag else None,
        )
    )

train_pipeline.append(
    dict(
        type=TrafficSignTarget,
        lidar_range=lidar_range,
        anno_name=anno_name,
        max_objects=max_objects,
        category2id=category2id,
        gt_augment_transform=True if augment_3d_flag else False,
    )
)

# 验证数据管道
val_pipeline = [
    dict(
        type=TrafficSignTarget,
        lidar_range=lidar_range,
        anno_name=anno_name,
        max_objects=max_objects,
        category2id=category2id,
        gt_augment_transform=False,
    )
]
```

#### 1.3 数据集配置
```python
# 帧采样器
def get_frame_sampler(down_sample_ratio):
    frame_sampler = dict(
        type=FrameSamplerTrafficSign,
        need_continuous=False,  # 交通标志通常不需要时序信息
        label_sample_scene={
            "intersection": 0.8,  # 重点采样路口场景
            "highway": 0.6,
            "urban": 0.4,
        },
        need_fix_sample=True if down_sample_ratio > 1 else False,
    )
    return frame_sampler

# 数据集配置
train_sample_config = dict(
    crop_frame_num=1,
    enable_temporal_sample=False,
)

val_sample_config = dict(
    sample_interval=1,
)

# 数据集获取函数
get_train_dataset = partial(
    model_base.get_dataset,
    pipeline=train_pipeline,
    frame_sampler=get_frame_sampler(1),
    sample_config=train_sample_config,
    length_for_rank_split=1000,
)

get_val_dataset = partial(
    model_base.get_test_dataset,
    pipeline=val_pipeline,
    frame_sampler=get_frame_sampler(1),
    sample_config=val_sample_config,
)
```

### 步骤2：创建目标处理模块

#### 2.1 创建目标处理类
```python
# projects/perception/target/traffic_sign_target.py
import numpy as np
import torch
from mmdet.core.bbox import BaseBox3D

class TrafficSignTarget:
    def __init__(self, lidar_range, anno_name, max_objects, category2id, gt_augment_transform=False):
        self.lidar_range = lidar_range
        self.anno_name = anno_name
        self.max_objects = max_objects
        self.category2id = category2id
        self.gt_augment_transform = gt_augment_transform
        
    def __call__(self, results):
        """处理交通标志标注数据"""
        # 获取标注信息
        annos = results.get(self.anno_name, {})
        
        if not annos:
            # 如果没有标注，返回空的目标
            return self._get_empty_targets(results)
        
        # 提取3D边界框信息
        gt_boxes_3d = []
        gt_labels = []
        gt_masks = []
        
        for anno in annos:
            if 'bbox_3d' in anno:
                bbox_3d = anno['bbox_3d']
                category = anno.get('category', 'unknown')
                
                if category in self.category2id:
                    # 转换为内部格式
                    gt_box = self._convert_bbox(bbox_3d)
                    gt_label = self.category2id[category]
                    
                    gt_boxes_3d.append(gt_box)
                    gt_labels.append(gt_label)
                    gt_masks.append(1.0)
        
        # 转换为tensor
        if gt_boxes_3d:
            gt_boxes_3d = torch.tensor(gt_boxes_3d, dtype=torch.float32)
            gt_labels = torch.tensor(gt_labels, dtype=torch.long)
            gt_masks = torch.tensor(gt_masks, dtype=torch.float32)
        else:
            gt_boxes_3d = torch.zeros((0, 7), dtype=torch.float32)  # x,y,z,w,l,h,theta
            gt_labels = torch.zeros((0,), dtype=torch.long)
            gt_masks = torch.zeros((0,), dtype=torch.float32)
        
        # 填充到最大数量
        if len(gt_boxes_3d) < self.max_objects:
            pad_size = self.max_objects - len(gt_boxes_3d)
            gt_boxes_3d = torch.cat([
                gt_boxes_3d,
                torch.zeros((pad_size, 7), dtype=torch.float32)
            ], dim=0)
            gt_labels = torch.cat([
                gt_labels,
                torch.zeros((pad_size,), dtype=torch.long)
            ], dim=0)
            gt_masks = torch.cat([
                gt_masks,
                torch.zeros((pad_size,), dtype=torch.float32)
            ], dim=0)
        
        # 更新结果
        results[f'{self.anno_name}_gt_boxes'] = gt_boxes_3d
        results[f'{self.anno_name}_gt_labels'] = gt_labels
        results[f'{self.anno_name}_gt_masks'] = gt_masks
        
        return results
    
    def _convert_bbox(self, bbox_3d):
        """转换3D边界框格式"""
        # 假设输入格式为: [x, y, z, dx, dy, dz, heading]
        # 输出格式为: [x, y, z, w, l, h, theta]
        x, y, z, dx, dy, dz, heading = bbox_3d
        return [x, y, z, dy, dx, dz, heading]  # w=dy, l=dx, h=dz
    
    def _get_empty_targets(self, results):
        """返回空的目标"""
        gt_boxes_3d = torch.zeros((self.max_objects, 7), dtype=torch.float32)
        gt_labels = torch.zeros((self.max_objects,), dtype=torch.long)
        gt_masks = torch.zeros((self.max_objects,), dtype=torch.float32)
        
        results[f'{self.anno_name}_gt_boxes'] = gt_boxes_3d
        results[f'{self.anno_name}_gt_labels'] = gt_labels
        results[f'{self.anno_name}_gt_masks'] = gt_masks
        
        return results
```

### 步骤3：创建模型头部

#### 3.1 创建检测头
```python
# projects/perception/model/head/traffic_sign_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import losses

class TrafficSignHead(nn.Module):
    def __init__(self, 
                 bev_h, bev_w, 
                 num_classes, num_query,
                 embed_dims, 
                 in_channels,
                 code_size=7,
                 use_aux_loss=True):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.code_size = code_size
        self.use_aux_loss = use_aux_loss
        
        # 查询嵌入
        self.query_embedding = nn.Embedding(num_query, embed_dims)
        
        # Transformer解码器
        self.decoder = self._build_decoder()
        
        # 分类头
        self.class_head = nn.Linear(embed_dims, num_classes)
        
        # 回归头
        self.bbox_head = nn.Linear(embed_dims, code_size)
        
        # 辅助分类头
        if use_aux_loss:
            self.aux_class_head = nn.Linear(embed_dims, num_classes)
        
        # 损失函数
        self.loss_cls = losses.FocalLoss(
            use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        )
        self.loss_bbox = losses.L1Loss(loss_weight=0.25)
        
    def _build_decoder(self):
        """构建Transformer解码器"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dims,
            nhead=8,
            dim_feedforward=self.embed_dims * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        return nn.TransformerDecoder(
            decoder_layer, num_layers=6
        )
    
    def forward(self, bev_features, metas=None):
        """前向传播"""
        batch_size = bev_features.size(0)
        
        # 展平BEV特征
        bev_flat = bev_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # 生成查询
        query = self.query_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer解码
        output = self.decoder(
            query.transpose(0, 1),  # [num_query, B, C]
            bev_flat.transpose(0, 1)  # [H*W, B, C]
        ).transpose(0, 1)  # [B, num_query, C]
        
        # 分类和回归
        cls_logits = self.class_head(output)  # [B, num_query, num_classes]
        bbox_pred = self.bbox_head(output)   # [B, num_query, code_size]
        
        # 辅助分类
        if self.use_aux_loss:
            aux_cls_logits = self.aux_class_head(output)
        else:
            aux_cls_logits = None
        
        return {
            'cls_logits': cls_logits,
            'bbox_pred': bbox_pred,
            'aux_cls_logits': aux_cls_logits,
            'query': query,
        }
    
    def loss(self, gt_boxes, gt_labels, gt_masks, predictions):
        """计算损失"""
        cls_logits = predictions['cls_logits']
        bbox_pred = predictions['bbox_pred']
        aux_cls_logits = predictions.get('aux_cls_logits')
        
        # 计算分类损失
        loss_cls = self.loss_cls(cls_logits, gt_labels, gt_masks)
        
        # 计算回归损失（只对正样本计算）
        pos_mask = gt_masks > 0
        if pos_mask.sum() > 0:
            pos_bbox_pred = bbox_pred[pos_mask]
            pos_gt_boxes = gt_boxes[pos_mask]
            loss_bbox = self.loss_bbox(pos_bbox_pred, pos_gt_boxes)
        else:
            loss_bbox = bbox_pred.sum() * 0
        
        # 辅助分类损失
        loss_aux = 0
        if aux_cls_logits is not None:
            loss_aux = self.loss_cls(aux_cls_logits, gt_labels, gt_masks)
        
        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_aux': loss_aux,
        }
    
    def get_results(self, predictions, metas=None):
        """获取推理结果"""
        cls_logits = predictions['cls_logits']
        bbox_pred = predictions['bbox_pred']
        
        # 应用sigmoid和softmax
        cls_scores = torch.sigmoid(cls_logits)
        
        # 后处理（NMS等）
        results = []
        batch_size = cls_scores.size(0)
        
        for i in range(batch_size):
            # 获取当前批次的结果
            batch_cls_scores = cls_scores[i]  # [num_query, num_classes]
            batch_bbox_pred = bbox_pred[i]   # [num_query, code_size]
            
            # 获取最大分数和对应类别
            max_scores, max_classes = torch.max(batch_cls_scores, dim=-1)
            
            # 简单的阈值过滤
            score_threshold = 0.3
            valid_mask = max_scores > score_threshold
            
            if valid_mask.sum() > 0:
                valid_boxes = batch_bbox_pred[valid_mask]
                valid_scores = max_scores[valid_mask]
                valid_classes = max_classes[valid_mask]
                
                results.append({
                    'boxes_3d': valid_boxes.cpu().numpy(),
                    'scores': valid_scores.cpu().numpy(),
                    'labels': valid_classes.cpu().numpy(),
                })
            else:
                results.append({
                    'boxes_3d': np.zeros((0, 7)),
                    'scores': np.zeros((0,)),
                    'labels': np.zeros((0,), dtype=np.int64),
                })
        
        return {
            'traffic_sign_results': results
        }
```

### 步骤4：创建数据采样器

#### 4.1 创建帧采样器
```python
# projects/perception/dataset/frame_sampler.py
class FrameSamplerTrafficSign:
    def __init__(self, 
                 need_continuous=False,
                 label_sample_scene=None,
                 need_fix_sample=False):
        self.need_continuous = need_continuous
        self.label_sample_scene = label_sample_scene or {}
        self.need_fix_sample = need_fix_sample
        
    def __call__(self, frame_infos, sample_config):
        """采样交通标志相关的帧"""
        if not self.need_continuous:
            # 不需要连续帧，随机采样
            if len(frame_infos) > 0:
                return [np.random.choice(frame_infos)]
            else:
                return []
        
        # 需要连续帧的采样逻辑
        sampled_frames = []
        # 实现具体的采样逻辑
        return sampled_frames
```

### 步骤5：创建评估指标

#### 5.1 创建评估指标类
```python
# projects/perception/callback/metric/traffic_sign_metric.py
import numpy as np
from leapai.callback.metric.base_metric import BaseMetric

class TrafficSignMetric(BaseMetric):
    def __init__(self, 
                 task_name,
                 annotation_name,
                 save_dir,
                 class_names,
                 distance_threshold=2.0,
                 score_threshold=0.3):
        super().__init__()
        self.task_name = task_name
        self.annotation_name = annotation_name
        self.save_dir = save_dir
        self.class_names = class_names
        self.distance_threshold = distance_threshold
        self.score_threshold = score_threshold
        
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.predictions = []
        self.ground_truths = []
    
    def process(self, predictions, ground_truth):
        """处理单个样本的预测和真值"""
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truth)
    
    def compute_metrics(self):
        """计算评估指标"""
        # 计算mAP、精确率、召回率等
        ap_per_class = []
        
        for class_id, class_name in enumerate(self.class_names):
            ap = self._compute_ap_for_class(class_id)
            ap_per_class.append(ap)
            print(f"{class_name} AP: {ap:.4f}")
        
        # 计算平均AP
        map_score = np.mean(ap_per_class)
        print(f"mAP: {map_score:.4f}")
        
        return {
            'mAP': map_score,
            'AP_per_class': dict(zip(self.class_names, ap_per_class))
        }
    
    def _compute_ap_for_class(self, class_id):
        """计算单个类别的AP"""
        # 实现AP计算逻辑
        # 这里简化实现，实际需要完整的IoU计算和AP计算
        return 0.0  # 占位符
    
    def save_results(self, results):
        """保存评估结果"""
        import json
        import os
        
        save_path = os.path.join(self.save_dir, f"{self.task_name}_results.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"结果已保存到: {save_path}")
```

### 步骤6：创建拓扑函数

#### 6.1 在配置文件中添加拓扑
```python
# 在 traffic_sign.py 中添加
# ----------------------------------Nodes-------------------------------------
nodes = {}

# BEV Neck
traffic_sign_neck = dict(
    type=Conv3x3Neck,
    input_c=embed_dims * 2,
    output_c=embed_dims,
)
neck_name = f"{task_name}_neck"
nodes[neck_name] = traffic_sign_neck

# 检测头
traffic_sign_head = dict(
    type=TrafficSignHead,
    bev_h=bev_h,
    bev_w=bev_w,
    num_classes=num_classes,
    num_query=256,
    embed_dims=embed_dims,
    in_channels=embed_dims,
    code_size=7,
    use_aux_loss=True,
)
head_name = f"{task_name}_head"
nodes[head_name] = traffic_sign_head

# ------------------------------topology--------------------------------------
def node_topology(state, model, batch, bev_feats, metas):
    """交通标志检测拓扑"""
    bev_neck = getattr(model, neck_name)
    bev_feats = bev_neck(bev_feats)
    
    head = getattr(model, head_name)
    head_pred = head(bev_feats, metas)
    
    if state == "train":
        gt = batch[anno_name]
        losses = head.loss(
            gt[f'{anno_name}_gt_boxes'],
            gt[f'{anno_name}_gt_labels'],
            gt[f'{anno_name}_gt_masks'],
            head_pred
        )
        return losses
    elif state == "val":
        preds = head.get_results(head_pred, metas)
        return preds
    else:
        raise NotImplementedError(state)

# ------------------------------metric-------------------------------------
def get_metric(test_set_name):
    """获取评估指标"""
    traffic_sign_metric = dict(
        type=TrafficSignMetric,
        task_name=test_set_name,
        annotation_name=anno_name,
        save_dir=save_root,
        class_names=class_names,
        distance_threshold=2.0,
        score_threshold=0.3,
    )
    return traffic_sign_metric
```

### 步骤7：集成到主配置

#### 7.1 修改主配置文件
```python
# 在主配置文件中添加交通标志任务
# 例如在 lpperception_current_hpa_step1.py 中添加：

# 启用交通标志任务
traffic_sign_task = True

# 更新多任务配置
multi_task_config = MAIN_CFG.multi_task_config.copy()
if traffic_sign_task:
    multi_task_config["traffic_sign"] = "projects/perception/traffic_sign.py"

# 更新数据集配置
if traffic_sign_task:
    train_set_info_path["traffic_sign"] = {
        "online": [
            "/path/to/traffic_sign_train_list.txt",
        ],
        "offline": "",
        "lmdb_path": "/path/to/traffic_sign_lmdb.txt",
    }
    
    val_set_info_path["traffic_sign"] = {
        "traffic_sign_test": dict(
            path="/path/to/traffic_sign_test_list.txt",
            lmdb_path="/path/to/traffic_sign_lmdb.txt",
        ),
    }

# 更新批次大小和采样配置
batch_sizes["traffic_sign"] = {"train": 8, "val": 1}
num_workers["traffic_sign"] = {"train": 4, "val": 4}
use_rank_split["traffic_sign"] = True
down_sample_ratio["traffic_sign"] = {"train": 1, "val": 1}
max_samples["traffic_sign"] = 100
```

### 步骤8：创建必要的目录和文件

#### 8.1 创建目录结构
```bash
# 创建必要的目录
mkdir -p projects/perception/model/head
mkdir -p projects/perception/target
mkdir -p projects/perception/callback/metric
mkdir -p projects/perception/dataset
```

#### 8.2 创建__init__.py文件
```python
# projects/perception/model/head/__init__.py
from .traffic_sign_head import TrafficSignHead

__all__ = ['TrafficSignHead']

# projects/perception/target/__init__.py
from .traffic_sign_target import TrafficSignTarget

__all__ = ['TrafficSignTarget']

# projects/perception/callback/metric/__init__.py
from .traffic_sign_metric import TrafficSignMetric

__all__ = ['TrafficSignMetric']
```

## 🧪 测试和验证

### 测试配置加载
```python
def test_traffic_sign_config():
    """测试交通标志配置加载"""
    import sys
    sys.path.append('/dahuafs/userdata/40359/Leapnet_master')
    
    from projects.perception.configs.traffic_sign import (
        class_names, category2id, get_train_dataset, 
        get_val_dataset, nodes, node_topology, get_metric
    )
    
    print("=== 交通标志任务配置测试 ===")
    print(f"类别数量: {len(class_names)}")
    print(f"类别: {class_names}")
    print(f"节点数量: {len(nodes)}")
    print("配置加载成功!")

test_traffic_sign_config()
```

### 测试数据流
```python
def test_data_flow():
    """测试数据流"""
    # 创建模拟数据
    batch_size = 2
    bev_h, bev_w = 128, 128
    embed_dims = 256
    
    # 模拟BEV特征
    bev_features = torch.randn(batch_size, embed_dims, bev_h, bev_w)
    
    # 模拟元数据
    metas = [{"scene_id": f"scene_{i}"} for i in range(batch_size)]
    
    # 测试模型头部
    from projects.perception.model.head.traffic_sign_head import TrafficSignHead
    
    head = TrafficSignHead(
        bev_h=bev_h, bev_w=bev_w,
        num_classes=5, num_query=256,
        embed_dims=embed_dims, in_channels=embed_dims
    )
    
    # 前向传播
    predictions = head(bev_features, metas)
    
    print("=== 数据流测试 ===")
    print(f"BEV特征形状: {bev_features.shape}")
    print(f"分类logits形状: {predictions['cls_logits'].shape}")
    print(f"边界框预测形状: {predictions['bbox_pred'].shape}")
    print("数据流测试成功!")

test_data_flow()
```

## 🚀 部署和使用

### 启动训练
```bash
# 设置环境变量
export LEAPAI_TASK_CONFIG="projects/perception/configs/lpperception_with_traffic_sign.py"
export RCNUM=1
export GPU_NUM=1
export my_debug="yes"

# 启动训练
python -m projects.perception.entry
```

### 监控训练
```python
# 监控交通标志任务的训练进度
def monitor_traffic_sign_training():
    """监控交通标志训练"""
    # 实现训练监控逻辑
    print("监控交通标志任务训练...")
    
monitor_traffic_sign_training()
```

## 📊 性能优化建议

### 1. 数据加载优化
- 使用LMDB加速数据读取
- 合理设置num_workers
- 启用数据预取

### 2. 模型优化
- 使用混合精度训练
- 实现模型编译优化
- 优化Transformer结构

### 3. 训练策略
- 使用学习率预热
- 实现梯度累积
- 添加正则化技术

## 🔧 常见问题解决

### 1. 导入错误
```python
# 确保所有模块正确导入
try:
    from projects.perception.configs.traffic_sign import *
    print("交通标志模块导入成功")
except ImportError as e:
    print(f"导入失败: {e}")
```

### 2. 配置冲突
```python
# 检查配置兼容性
def check_config_compatibility():
    """检查配置兼容性"""
    # 实现配置检查逻辑
    pass

check_config_compatibility()
```

### 3. 内存问题
```python
# 优化内存使用
def optimize_memory():
    """优化内存使用"""
    torch.cuda.empty_cache()
    # 其他内存优化策略

optimize_memory()
```

## 🎯 总结

通过本指南，您已经学会了：

1. **任务分析**: 如何分析新任务的需求
2. **配置创建**: 如何创建任务配置文件
3. **模块实现**: 如何实现数据处理、模型和评估模块
4. **系统集成**: 如何将新任务集成到主框架
5. **测试验证**: 如何测试和验证新任务
6. **部署使用**: 如何启动和监控新任务训练

## 📚 扩展建议

1. **添加更多类别**: 扩展交通标志类别
2. **改进模型结构**: 使用更先进的检测架构
3. **多模态融合**: 结合更多传感器数据
4. **时序建模**: 添加时序信息处理
5. **部署优化**: 针对推理场景优化

---

**注意**: 本指南提供了一个完整的添加新任务的流程示例。在实际应用中，您需要根据具体任务需求调整实现细节。
