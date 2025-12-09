# LeapAI Step2 Dynamic配置详细网络架构分析

## 📋 配置环境

### 启动配置
```json
{
    "name": "bev test debug",
    "program": "/opt/conda/bin/torchrun",
    "env": {"LEAPAI_TASK_CONFIG": "projects/perception/configs/lpperception_current_hpa_step2_dynamic.py"},
    "args": ["--nproc_per_node=1", "--master_port=21212", "tools/main.py", "--config", "projects/perception/entry.py", "--state", "val", "--ckpt", "/path/to/step2.ckpt"]
}
```

### 任务配置
```python
# Step2 Dynamic专用配置
job_name = "lphpa_v3.0_step2_dynamic"
enable_lidar = True
use_dynamic_outputs = True
dynamic_task = True          # 启用动态任务
static_task = False         # 关闭静态任务
occ_task = False           # 关闭占用任务
attr_task = False          # 关闭属性任务

# 时序配置
enable_dynamic_temporal = True   # 启用动态时序融合
enable_static_temporal = False    # 关闭静态时序
enable_occ_temporal = False       # 关闭占用时序

# 训练配置
train_steps = 0                    # 无基础训练
finetune_steps = 75000            # 微调步数
max_steps = 75000                  # 总步数
```

## 🎯 与其他配置的关键区别

### Step1 vs Step2 Dynamic
```python
# Step1配置 (多任务联合训练)
dynamic_task = True
static_task = True
enable_dynamic_temporal = False
train_steps = 125000
finetune_steps = 0

# Step2 Dynamic配置 (动态任务微调)
dynamic_task = True
static_task = False
enable_dynamic_temporal = True
train_steps = 0
finetune_steps = 75000
```

### Step2 Static vs Step2 Dynamic
```python
# Step2 Static配置
dynamic_task = False
static_task = True
enable_dynamic_temporal = False
enable_static_temporal = True

# Step2 Dynamic配置  
dynamic_task = True
static_task = False
enable_dynamic_temporal = True
enable_static_temporal = False
```

## 🏗️ Step2 Dynamic专用网络结构

### 🔄 动态时序融合模块 (核心组件)
```python
# 时序管理器配置
topk_query = 128                    # Top-K查询数量
queue_length = 4                    # 时序队列长度

Temporal_manager:
├── 功能: 管理动态目标的时序信息
├── 队列长度: 4帧历史
├── Top-K选择: 128个最高置信度查询
├── 速度估计: 3D速度向量 (vx, vy, vz)
├── 位置跟踪: 3D参考点
├── 时间戳管理: 微秒级时间精度
└── 坐标变换: leapego2global矩阵
```

### 🧠 冻结策略 (关键优化)
```python
freeze_module_schedules = {
    "stage1_backbone0": train_steps,    # 冻结ResNet34 Group1
    "stage1_backbone1": train_steps,    # 冻结ResNet34 Group2
    "stage1_backbone2": train_steps,    # 冻结ResNet34 Group3
    "stage1_backbone3": train_steps,    # 冻结ResNet34 Group4
    "stage1_neck0": train_steps,        # 冻结FPN Neck Group1
    "stage1_neck1": train_steps,        # 冻结FPN Neck Group2
    "stage1_neck2": train_steps,        # 冻结FPN Neck Group3
    "stage1_neck3": train_steps,        # 冻结FPN Neck Group4
    "view_transfer": train_steps,        # 冻结BEV变换
    "lidar_vfe": train_steps,           # 冻结LiDAR特征提取
    "lidar_middle_encoder": train_steps,  # 冻结LiDAR中间编码器
    "lidar_bev_backbone": train_steps,   # 冻结LiDAR BEV骨干
    "fuser": train_steps,               # 冻结多模态融合器
}
```

### 🎯 可训练模块 (仅训练这些)
```python
trainable_modules = [
    "bev_dynamic_head",           # 动态检测头
    "bev_dynamic_neck",           # 动态颈部网络
]
```

## 📊 数据流动与Shape变化

### 动态时序数据流
```
Step 1: 输入数据预处理
├── 相机图像: [B, 7, 3, 512, 960] → 150MB
├── LiDAR点云: 变长 → 体素化 → 61MB
├── 标注数据: 动态检测框 + 速度 + 遮挡
└── 元数据: 时间戳 + 位姿矩阵

Step 2: 特征提取 (冻结)
├── 相机特征: 4×ResNet34V2 → [B, 7, 256, H/8, W/8]
├── FPN颈部: 多尺度融合 → [B, 7, 256, H/8, W/8]
├── BEV变换: IPM + 可变形注意力 → [B, 256, 112, 128]
├── LiDAR特征: VFE + 中间编码器 → [B, 256, 112, 128]
└── 多模态融合: 相机+LiDAR → [B, 256, 112, 128]

Step 3: 动态颈部处理 (可训练)
├── 输入特征: [B, 256, 112, 128]
├── ConvResBlockNeck: 残差卷积块
├── 特征增强: [B, 256, 112, 128] → [B, 512, 112, 128]
└── 展平处理: [B, 512, 112, 128] → [B, 14336, 512]

Step 4: 时序查询管理
├── 当前查询: [B, 384, 256] (384个learnable query)
├── 历史查询: [B, 128×3, 256] (128×3个历史query)
├── 查询拼接: [B, 384+384, 256] → [B, 768, 256]
├── 参考点: [B, 384, 3] + [B, 384, 3] (历史)
└── 速度信息: [B, 384, 3] (历史速度)

Step 5: Transformer解码器 (可训练)
├── 自注意力: 当前查询 + 历史查询
├── 交叉注意力: 查询 ↔ BEV特征
├── 位置编码: 3D位置编码器
├── 多层处理: 3层StreamTransformerLayer
└── 输出特征: [3, B, 384, 256]

Step 6: 动态检测头 (可训练)
├── 分类分支: [3, B, 384, 8] → 8类动态目标
├── 回归分支: [3, B, 384, 17] → 3D边界框 + 速度 + 遮挡
├── 角度分箱: [3, B, 384, 8] → 8个角度区间
├── 遮挡分类: [3, B, 384, 1] → 遮挡状态
└── 最终输出: 检测结果 + 时序信息
```

### 内存占用分析
```python
# 基础内存 (与Step1相同)
Base Memory: ~4.3GB per GPU
├── 输入数据: ~211MB
├── 特征提取: ~3.9GB
└── 基础输出: ~150MB

# 动态时序模块额外内存
Dynamic Temporal Memory: ~600MB per GPU
├── 查询队列: 4 × 128 × 256 × 4bytes ≈ 500KB
├── 参考点队列: 4 × 128 × 3 × 4bytes ≈ 6KB
├── 速度队列: 4 × 128 × 3 × 4bytes ≈ 6KB
├── 时间戳队列: 4 × 128 × 8bytes ≈ 4KB
├── 位姿矩阵: 4 × 128 × 4 × 4 × 4bytes ≈ 32KB
├── Transformer计算: ~100MB
└── 时序对齐计算: ~500MB

# 总内存占用
Total Memory: ~4.9GB per GPU
```

## 🎯 网络输出详细说明

### 动态检测输出 (时序增强)
```python
# 基础检测输出
Classification: [B, 384, 8]
├── Classes: 8类动态目标
├── Class Names: 
│   ├── 0: car (汽车)
│   ├── 1: truck (卡车)
│   ├── 2: bus (公交车)
│   ├── 3: person (行人)
│   ├── 4: non_motor (非机动车)
│   ├── 5: riderless_non_motor (无人非机动车)
│   ├── 6: barrier (障碍物)
│   └── 7: pillar (柱子)
└── Confidence: Sigmoid激活

3D Bounding Box: [B, 384, 11]
├── Center: (x, y, z) - BEV坐标系中心点
├── Size: (l, w, h) - 长宽高
├── Yaw: θ - 朝向角
├── Velocity: (vx, vy, vz) - 3D速度向量
└── Format: 归一化坐标 + 实际速度

Angle Binning: [B, 384, 8]
├── Bins: 8个角度区间 (45°每个)
├── Overlap: 1/36 重叠区间
├── Purpose: 精确角度预测
└── Output: 软最大值分布

Occlusion Classification: [B, 384, 1]
├── Classes: 4类遮挡状态
│   ├── 0: NoOccluded (无遮挡)
│   ├── 1: SlightlyOccluded (轻微遮挡)
│   ├── 2: PartlyOccluded (部分遮挡)
│   └── 3: HeavilyOccluded (严重遮挡)
└── Purpose: 遮挡感知与处理
```

### 时序信息输出
```python
# 时序管理输出
Temporal Query Info:
├── Current Query: [B, 384, 256] - 当前帧查询
├── Historical Query: [B, 384, 256] - 历史帧查询
├── Reference Points: [B, 384, 3] - 3D参考点
├── Velocity Vectors: [B, 384, 3] - 速度向量
├── Timestamps: [B, 384] - 时间戳
├── Transform Matrices: [B, 384, 4, 4] - 坐标变换矩阵
└── Object IDs: [B, 384] - 目标ID跟踪

# 时序对齐输出
Temporal Alignment:
├── Prev2Curr Matrix: [B, 4, 4] - 历史到当前变换
├── Time Intervals: [B] - 时间间隔(秒)
├── Motion Compensation: 速度×时间间隔
├── Coordinate Transform: BEV坐标对齐
└── Clipping: [0, 1]范围限制
```

### 验证模式输出
```python
# 验证时的特殊输出
Validation Outputs:
├── Detection Results: NMS后的检测框
├── Confidence Scores: 置信度分数
├── Class Labels: 类别标签
├── 3D Boxes: 3D边界框坐标
├── Velocities: 速度估计
├── Occlusion States: 遮挡状态
├── Object Tracks: 目标轨迹
└── Temporal Consistency: 时序一致性分数
```

## 🚀 推理与部署

### 验证模式配置
```python
# 验证时的特殊处理
state = "val"
eval_with_visualize = True  # 启用可视化
eval_instance = False       # 关闭实例评估

# 批次大小
batch_sizes = {
    "dynamic": {"train": 16, "val": 1},  # 验证时batch_size=1
}
```

### 可视化输出
```python
# BEV可视化配置
draw_dynamic_cfg = dict(
    dynamic_conf=0.3,               # 动态目标置信度
    draw_velo=True,                 # 绘制速度向量
    draw_occlusion=True,            # 绘制遮挡状态
    draw_trajectory=True,            # 绘制轨迹
    valid_range=[-150, 20, -25, 25],  # 可视化范围
    coords=[-40, 62.4, -44.8, 44.8],  # BEV坐标范围
    label_h=112,                    # 标签高度
    label_w=128,                    # 标签宽度
    max_objects=100,                # 最大显示目标数
)

# 时序可视化
temporal_visualization = {
    "show_history": True,          # 显示历史轨迹
    "show_velocity": True,         # 显示速度向量
    "show_occlusion": True,        # 显示遮挡状态
    "history_frames": 3,           # 历史帧数量
    "trajectory_length": 10,       # 轨迹长度
    "velocity_scale": 5.0,         # 速度向量缩放
}
```

## 📈 性能优化策略

### 内存优化
```python
# 冻结策略
freeze_modules = [
    "stage1_backbone*",      # 冻结所有骨干网络
    "stage1_neck*",         # 冻结所有颈部网络
    "view_transfer",        # 冻结视图变换
    "lidar_*",              # 冻结LiDAR模块
    "fuser",               # 冻结融合器
]

# 可训练模块
trainable_modules = [
    "bev_dynamic_head",           # 动态检测头
    "bev_dynamic_neck",           # 动态颈部网络
]
```

### 计算优化
```python
# 混合精度
use_backbone_amp = True  # 骨干网络AMP

# 梯度累积
accumulate_grad_batches = 1

# 学习率缩放
group_lr_scale = {
    "stage1_backbone": 1.0,  # 冻结，实际不更新
    "stage1_neck": 1.0,       # 冻结，实际不更新
}

# 时序优化
temporal_optimization = {
    "topk_selection": True,      # Top-K查询选择
    "query_rearrangement": True, # 查询重排
    "velocity_zeroing": True,    # SOD速度置零
    "random_sampling": True,     # 随机采样策略
}
```

## 🎯 关键技术特点

### 动态时序融合优势
1. **目标跟踪**: 128个Top-K查询的持续跟踪
2. **运动估计**: 3D速度向量的精确估计
3. **时序一致性**: 多帧信息融合提升稳定性
4. **遮挡处理**: 4级遮挡分类与处理
5. **坐标对齐**: 精确的时空坐标变换

### 专注动态优化
1. **任务专注**: 关闭静态任务，专注动态检测
2. **参数高效**: 冻结大部分网络，只训练关键模块
3. **时序增强**: 利用历史信息提升检测性能
4. **快速收敛**: 微调阶段快速达到最优性能

### 智能查询管理
```python
# Top-K查询选择策略
Query Selection Strategy:
├── 置信度排序: 选择最高置信度的128个查询
├── 随机采样: 50%概率使用增强采样
├── 分层选择: Top-20 + 随机113的组合策略
├── 查询重排: 按索引位置重新排列
└── 历史融合: 当前查询与历史查询的智能融合

# 速度处理策略
Velocity Processing:
├── SOD速度置零: 静态障碍物速度设为0
├── 运动补偿: 基于速度的位姿预测
├── 时序对齐: 历史位置到当前帧的变换
├── 坐标裁剪: 确保位置在有效范围内
└── 速度传播: 历史速度信息的利用
```

## 📊 损失函数设计

### 多任务损失权重
```python
# 损失权重配置
task_loss_weights = {
    "dynamic": 1.25,  # 动态任务权重(时序增强)
}

# 动态检测损失
Dynamic Loss Components:
├── Classification Loss: Focal Loss (α=0.25, γ=2.0)
├── Bounding Box Loss: L1 Loss
├── Angle Binning Loss: CrossEntropy Loss
├── Occlusion Loss: Focal Loss
├── Instance Weights: 距离加权
└── Velocity Loss: 遮挡加权

# 距离加权策略
Distance Weighting:
├── VRU近处: 0-6m → 4.0倍权重
├── VRU中距离: 6-12m → 2.0倍权重
├── MOD近处: 0-6m → 4.0倍权重
├── MOD中距离: 6-12m → 2.0倍权重
└── SOD权重: 根据距离递减
```

这个Step2 Dynamic配置代表了动态检测任务的专门优化阶段，通过时序融合和智能查询管理，实现了高精度的动态目标感知与跟踪。
