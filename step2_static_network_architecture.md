# LeapAI Step2 Static配置详细网络架构

## 🎯 配置环境分析

基于您的新配置：
```json
{
    "name": "bev test debug",
    "program": "/opt/conda/bin/torchrun",
    "env": {"LEAPAI_TASK_CONFIG": "projects/perception/configs/lpperception_current_hpa_step2_static.py"},
    "args": ["--nproc_per_node=1", "--master_port=21212", "tools/main.py", "--config", "projects/perception/entry.py", "--state", "val", "--ckpt", "/path/to/step2.ckpt"]
}
```

## 🏗️ 网络架构关键变化

### 与Step1的主要区别
```python
# Step1配置 (训练)
dynamic_task = True      # 启用动态任务
static_task = True       # 启用静态任务
enable_static_temporal = False  # 关闭时序

# Step2配置 (静态微调)
dynamic_task = False     # 关闭动态任务
static_task = True       # 启用静态任务
enable_static_temporal = True   # 启用时序
```

### 训练阶段配置
```python
# Step2是静态任务的微调阶段
num_train_cases = 0           # 不进行训练
num_finetune_cases = 120000   # 微调12万cases
train_steps = 0               # 训练步数为0
finetune_steps = 75000         # 微调步数
max_steps = 75000              # 总步数=微调步数

# 预训练权重
float_pretrain = "/annotationdata/.../step1.ckpt"  # 从Step1加载
```

## 📊 网络结构详细分析

### 🎯 启用的模块
✅ **静态任务**: 主要训练目标  
✅ **LiDAR融合**: 多模态感知  
✅ **时序处理**: 静态任务时序融合  
✅ **可视化**: 评估时启用可视化  
❌ **动态任务**: 关闭以专注静态  
❌ **占用任务**: 关闭  
❌ **属性任务**: 关闭  

### 🔄 时序处理模块

#### 静态时序配置
```python
static_temporal_config = dict(
    enable_temporal=True,           # 启用时序
    fusion_conv="VGGBlock",        # VGG块融合
    grid_sample_mode="nearest",      # 最近邻采样
    visualize_interval=8888,       # 可视化间隔
    debug_dir=None,
)

static_temporal_sample_cfg = dict(
    mode="sample_by_timestamp",     # 按时间戳采样
    intervals=[0.7, 1.9, 3.3],   # 时间间隔 [秒]
    match_tol=0.1,               # 匹配容差
    interval_variable=False,         # 固定间隔
    max_capacity=40,              # 最大容量40帧
)
```

#### 时序融合网络结构
```
HistoryFeatureManager:
├── Function: 管理历史特征队列
├── Queue Length: 4 (当前帧 + 3历史帧)
├── Intervals: [0.7s, 1.9s, 3.3s]
├── Max Capacity: 40 frames
├── Match Tolerance: 0.1s
└── Output: [prev_feat_list, feature_warp_matrix]

TemporalFusion:
├── Input: 当前BEV特征 + 历史特征
├── Fusion Conv: VGGBlock
│   ├── Conv3x3: 128→128
│   ├── ReLU
│   ├── Conv3x3: 128→128
│   └── ReLU
├── Grid Sample: 特征对齐
├── Output: 融合后的BEV特征
└── Parameters: ~100,000
```

### 🧠 冻结的模块

#### 骨干网络冻结
```python
freeze_module_schedules = {
    "stage1_backbone0": 75000,    # ResNet34 Group1
    "stage1_backbone1": 75000,    # ResNet34 Group2  
    "stage1_backbone2": 75000,    # ResNet34 Group3
    "stage1_backbone3": 75000,    # ResNet34 Group4
    "stage1_neck0": 75000,        # FPN Neck Group1
    "stage1_neck1": 75000,        # FPN Neck Group2
    "stage1_neck2": 75000,        # FPN Neck Group3
    "stage1_neck3": 75000,        # FPN Neck Group4
    "view_transfer": 75000,        # BEV变换
}
```

#### LiDAR模块冻结
```python
if enable_lidar:
    freeze_module_schedules.update({
        "lidar_vfe": 75000,           # LiDAR特征提取
        "lidar_middle_encoder": 75000,  # LiDAR中间编码器
        "lidar_bev_backbone": 75000,   # LiDAR BEV骨干
        "fuser": 75000,               # 多模态融合器
    })
```

### 🎯 静态任务专用网络

#### 静态分割头增强
```python
# 基础静态头 (与Step1相同)
static_head = dict(
    type=StaticSegHead,
    # ... 基础配置
)

# 时序融合头 (Step2新增)
static_temporal_fusion = dict(
    type=TemporalFusion,
    embed_dims=128,
    queue_length=4,  # 当前+3历史
    pc_range=[-20.8, -22.4, -3.0, 62.4, 22.4, 5.0],
    history_featmanager=dict(
        type=HistoryFeatureManager,
        mode="sample_by_timestamp",
        intervals=[0.7, 1.9, 3.3],
        match_tol=0.1,
        max_capacity=40,
    ),
)
```

## 📊 数据流动与Shape变化

### 时序数据流
```
Step 1: 历史特征管理
├── 输入: 当前BEV特征 [16, 128, 56, 104]
├── 历史队列: 最多40帧历史
├── 时间匹配: 按时间戳匹配历史帧
├── 特征对齐: grid_sample变换
└── 输出: [prev_feat_list, feature_warp_matrix]

Step 2: 时序融合
├── 当前特征: [16, 128, 56, 104]
├── 历史特征: [[16, 128, 56, 104] × 3]
├── 融合操作: VGGBlock卷积融合
├── 输出特征: [16, 128, 56, 104]
└── 时序信息: 传递给下游任务

Step 3: 静态分割
├── 输入: 时序融合特征 [16, 128, 56, 104]
├── 上采样: [16, 128, 56, 104] → [16, 128, 224, 416]
├── 多任务分割:
│   ├── 车道线: [16, 9, 224, 416]
│   ├── 道路: [16, 4, 224, 416]
│   └── 实例: [16, 100, 6]
└── 后处理: NMS + 阈值过滤
```

### 内存占用变化
```python
# 基础内存 (与Step1相同)
Base Memory: ~4.3GB per GPU
├── 输入数据: ~211MB
├── 特征提取: ~3.9GB
└── 基础输出: ~150MB

# 时序模块额外内存
Temporal Memory: ~800MB per GPU
├── 历史特征队列: 40 × 16 × 128 × 56 × 104 × 4bytes ≈ 600MB
├── 变换矩阵: 40 × 16 × 1 × 4 × 2 × 4bytes ≈ 80MB
├── 融合计算: ~100MB
└── 时序输出: ~20MB

# 总内存占用
Total Memory: ~5.1GB per GPU
```

## 🔧 训练配置详解

### 优化器设置
```python
# 微调阶段学习率
finetune_lr = 2e-4 * sqrt(num_gpus / 8)  # 与Step1相同

# 损失权重调整
static_loss_weight = 1.0  # 时序开启时权重增加 (Step1: 0.67)

# 学习率调度
milestones, lr_list = base.get_mutistep_gamma_lr(
    multi_lr_milestones=dict(
        train=[0, 0],  # 无训练阶段
        finetune=[
            0.65 * finetune_steps,  # 48,750步
            0.9 * finetune_steps    # 67,500步
        ],
    ),
    train_steps=0,
    finetune_steps=75000,
    gamma=0.1,
    lr=dict(train=2e-4, finetune=2e-4),
)
```

### 数据集配置
```python
# 微调数据集
train_set_info_path["static"] = {
    "online": [
        "/dahuafs/groupdata/share/perception/dataset/Static_HPA/20251203/BEVStatic_Train_HPA_Parking_2312214_1088_train.txt",
        "/dahuafs/groupdata/share/perception/dataset/Static_HPA/20251203/BEVStatic_Train_HPA_Parking_2312214_1088_train.txt",
        "/dahuafs/groupdata/share/perception/dataset/Static_HPA/20251203/BEVStatic_Train_HPA_Parking_2312214_1088_train.txt",
        "/dahuafs/groupdata/share/perception/dataset/Static_HPA/20251203/BEVStatic_Train_HPA_Parking_2312214_504.txt",
    ],
    "lmdb_path": "/dahuafs/groupdata/Cameraalgorithm/tmp/szh/code_Git_2025_07/leapnet_March/lmdb_1783288.txt",
}

# 验证数据集
val_set_info_path["static"] = {
    "EE3.5_B10_112": dict(
        path="/dahuafs/groupdata/share/perception/dataset/Static_HPA/20251203/BEVStatic_Train_HPA_Parking_2312214_1088_test.txt",
        lmdb_path="/dahuafs/groupdata/Cameraalgorithm/tmp/szh/code_Git_2025_07/leapnet_March/lmdb_1783288.txt",
    ),
    "Entrance_Exit": dict(
        path="/dahuafs/groupdata/share/perception/dataset/Static_HPA/20251203/BEVStatic_Test_HPA_Parking_2311399_12.txt",
        lmdb_path="/dahuafs/groupdata/Cameraalgorithm/tmp/szh/code_Git_2025_07/leapnet_March/lmdb_1783288.txt",
    ),
}
```

## 🎯 网络输出详细说明

### 静态分割输出 (时序增强)
```python
# 基础分割输出
Lane Segmentation: [16, 9, 224, 416]
├── Classes: 9类车道线
├── Resolution: 0.2m × 0.2m
├── Coverage: 83.2m × 44.8m
└── Classes: [SolidLine, DoubleSolidLine, DashedLine, DoubleDashedLine, 
              RightSolidLeftDashed, LeftSolidRightDashed, WideSolidLine, 
              WideDashedLine, ShortDashedLine]

Road Segmentation: [16, 4, 224, 416]
├── Classes: 4类道路元素
├── Resolution: 0.2m × 0.2m
├── Coverage: 83.2m × 44.8m
└── Classes: [Wall, Curb, Lane, SpeedBump, GroundSigns]

Instance Detection: [16, 100, 6]
├── Max Instances: 100
├── Format: [offset_x, offset_y, w, h, angle, class]
├── Coordinate: BEV
└── Units: meters, radians

# 时序信息输出
Temporal Features: [16, 3, 128, 56, 104]
├── Historical Frames: 3个历史帧
├── Time Intervals: [0.7s, 1.9s, 3.3s]
├── Feature Alignment: grid_sample变换
└── Fusion Weights: VGGBlock学习权重
```

### 系列评估输出
```python
# 系列评估配置
static_series_eval = True
series_eval_conf = [0.9, 0.85, 0.8, 0.75, 0.7]
series_dis_thresh = 0.25

# 评估范围
series_eval_range = {
    "all_range": [-22.4, 22.4, -20.8, 62.4],      # 全范围
    "main_range": [-8.0, 8.0, -20.0, 120.0],     # 主范围
    "main_near_range": [-8.0, 8.0, 0.0, 30.0],   # 近范围
    "main_middle_range": [-8.0, 8.0, 30.0, 60.0],  # 中范围
    "main_far_range": [-8.0, 8.0, 60.0, 120.0],    # 远范围
}

# 子类别评估
series_sub_cls = {
    "lanes": {
        "_marking_type/lane": [
            "Other", "SolidLine", "DoubleSolidLine", "DashedLine",
            "DoubleDashedLine", "RightSolidLeftDashed", 
            "LeftSolidRightDashed", "WideSolidLine", 
            "WideDashedLine", "ShortDashedLine"
        ],
    },
}
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
    "static": {"train": 16, "val": 1},  # 验证时batch_size=1
}

# 数据加载
num_workers = {
    "static": {"train": 4, "val": 4},
}
```

### 可视化输出
```python
# BEV可视化配置
draw_static_cfg = dict(
    lane_conf=0.75,              # 车道线置信度
    road_conf=0.75,              # 道路置信度
    freespace_conf=0.5,           # 自由空间置信度
    stopline_conf=0.6,           # 停止线置信度
    crosswalk_conf=0.9,           # 人行横道置信度
    valid_range=[-150, 20, -25, 25],  # 可视化范围
    coords=[-22.4, 62.4, -20.8, 22.4],  # BEV坐标范围
    label_h=224,                 # 标签高度
    label_w=416,                 # 标签宽度
    querylane_points_num=10,        # 查询车道点数
    querylane_threshold=0.3,        # 查询车道阈值
)

# 时序可视化
temporal_visualization = {
    "show_history": True,          # 显示历史帧
    "show_fusion": True,          # 显示融合结果
    "history_frames": 3,           # 历史帧数量
    "fusion_weights": True,        # 显示融合权重
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
    "bev_static_head",           # 静态分割头
    "bev_static_temporal_fusion", # 时序融合模块
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
```

## 🎯 关键技术特点

### 时序融合优势
1. **时间一致性**: 利用历史信息提升分割一致性
2. **噪声抑制**: 多帧融合减少单帧噪声影响
3. **运动补偿**: 特征对齐处理车辆运动
4. **长期记忆**: 40帧历史提供长期上下文

### 专注静态优化
1. **任务专注**: 关闭动态任务，专注静态分割
2. **参数高效**: 冻结大部分网络，只训练关键模块
3. **快速收敛**: 微调阶段快速收敛到最优性能
4. **稳定训练**: 减少多任务干扰

这个Step2 Static配置代表了静态分割任务的专门优化阶段，通过时序融合和模块冻结策略，实现了高精度的静态元素感知。
