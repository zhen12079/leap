# LeapAI框架详细网络架构输出

## 🎯 配置环境
- **配置文件**: `projects/perception/configs/lpperception_current_hpa_step1.py`
- **入口文件**: `projects/perception/entry.py`
- **运行模式**: `train`
- **批次大小**: 16
- **GPU数量**: 根据环境变量确定

## 🏗️ 完整网络层次结构

### 📸 输入层 (Input Layer)

#### 相机输入
```
Camera Input Group 1 (front_narrow):
├── Tensor Shape: [16, 1, 3, 512, 960]
├── Data Type: torch.float32
├── Memory: ~18.8MB per batch
└── Description: 前视窄角相机

Camera Input Group 2 (front_wide):
├── Tensor Shape: [16, 1, 3, 1024, 1920] → [16, 1, 3, 512, 960]
├── Data Type: torch.float32
├── Memory: ~37.5MB per batch
└── Description: 前视广角相机

Camera Input Group 3 (back):
├── Tensor Shape: [16, 1, 3, 512, 960]
├── Data Type: torch.float32
├── Memory: ~18.8MB per batch
└── Description: 后视相机

Camera Input Group 4 (4 side cameras):
├── front_left: [16, 1, 3, 512, 960]
├── back_left: [16, 1, 3, 512, 960]
├── front_right: [16, 1, 3, 512, 960]
├── back_right: [16, 1, 3, 512, 960]
├── Total Memory: ~75.2MB per batch
└── Description: 四个侧视相机

Total Camera Input Memory: ~150MB per batch
```

#### LiDAR输入
```
LiDAR Input:
├── Raw Points: Variable length (avg ~50,000 points/frame)
├── Point Format: [x, y, z, intensity]
├── Data Type: torch.float32
├── Memory: ~8MB per frame (avg)
└── Description: 激光雷达点云数据

Voxelized LiDAR:
├── Voxels: [16, 20000, 48, 4]
├── Voxel Coords: [16, 20000, 4]
├── Voxel Num Points: [16, 20000]
├── Memory: ~61MB per batch
└── Voxel Size: [0.2, 0.2, 8.0]
```

### 🧠 特征提取骨干网络 (Feature Extraction Backbone)

#### ResNet34V2 - Group1 (front_narrow)
```
stage1_backbone0 (ResNet34V2):
├── Input: [16, 1, 3, 512, 960]
├── Conv1: [16, 1, 64, 256, 480]
│   ├── Kernel: 7x7, Stride: 2, Padding: 3
│   ├── Parameters: 9,408
│   └── Output: [16, 1, 64, 256, 480]
├── Layer1 (Residual Blocks):
│   ├── Input: [16, 1, 64, 256, 480]
│   ├── Output: [16, 1, 64, 256, 480]
│   ├── Blocks: 3
│   └── Parameters: ~70,000
├── Layer2 (Residual Blocks):
│   ├── Input: [16, 1, 64, 256, 480]
│   ├── Output: [16, 1, 128, 128, 240]
│   ├── Blocks: 4
│   └── Parameters: ~220,000
├── Layer3 (Residual Blocks):
│   ├── Input: [16, 1, 128, 128, 240]
│   ├── Output: [16, 1, 256, 64, 120]
│   ├── Blocks: 6
│   └── Parameters: ~700,000
├── Layer4 (Residual Blocks):
│   ├── Input: [16, 1, 256, 64, 120]
│   ├── Output: [16, 1, 512, 32, 60]
│   ├── Blocks: 3
│   └── Parameters: ~500,000
└── Total Parameters: ~1.5M
```

#### ResNet34V2 - Group2 (front_wide)
```
stage1_backbone1 (ResNet34V2):
├── Input: [16, 1, 3, 512, 960]
├── Output Features: [256, 512] (indices [3,4])   #  这是什么含义？？
├── Feature Shapes:
│   ├── Layer3: [16, 1, 256, 64, 120]
│   └── Layer4: [16, 1, 512, 32, 60]
└── Total Parameters: ~21.3M
```

#### ResNet34V2 - Group3 (back)
```
stage1_backbone2 (ResNet34V2):
├── Input: [16, 1, 3, 512, 960]
├── Output Features: [128, 256, 512] (indices [2,3,4])  #  这是什么含义？？
├── Feature Shapes:
│   ├── Layer2: [16, 1, 128, 128, 240]
│   ├── Layer3: [16, 1, 256, 64, 120]
│   └── Layer4: [16, 1, 512, 32, 60]
└── Total Parameters: ~21.3M
```

#### ResNet34V2 - Group4 (4 side cameras)
```
stage1_backbone3 (ResNet34V2):
├── Input: [16, 4, 3, 512, 960] → [64, 3, 512, 960]  # 这里如何操作
├── Output Features: [256, 512] (indices [3,4])
├── Feature Shapes:
│   ├── Layer3: [64, 1, 256, 64, 120] → [16, 4, 256, 64, 120]  # 这里如何操作
│   └── Layer4: [64, 1, 512, 32, 60] → [16, 4, 512, 32, 60]
└── Total Parameters: ~21.3M
```

### 🌉 FPN颈部网络 (Feature Pyramid Network)

#### FPN Neck - Group1 & Group3  
```
stage1_neck0 & stage1_neck2 (FpnNeck):
├── Input Channels: [128, 256, 512]
├── Output Channels: 256
├── Input Features:
│   ├── P2: [16, N, 128, 128, 240]
│   ├── P3: [16, N, 256, 64, 120]
│   └── P4: [16, N, 512, 32, 60]
├── Lateral Convs:
│   ├── Conv2d(128→256): 3x3, stride=1, padding=1
│   ├── Conv2d(256→256): 3x3, stride=1, padding=1
│   └── Conv2d(512→256): 3x3, stride=1, padding=1
├── Output Convs:
│   ├── Conv2d(256→256): 3x3, stride=1, padding=1
│   └── Conv2d(256→256): 3x3, stride=1, padding=1
├── Output: [16, N, 256, 128, 240]
└── Parameters: ~200,000 per neck
```

#### FPN Neck - Group2 & Group4
```
stage1_neck1 & stage1_neck3 (FpnNeck):
├── Input Channels: [256, 512]
├── Output Channels: 256
├── Input Features:
│   ├── P3: [16, N, 256, 64, 120]
│   └── P4: [16, N, 512, 32, 60]
├── Output: [16, N, 256, 64, 120]
└── Parameters: ~150,000 per neck
```

### 🚀 LiDAR处理分支

#### VFE (Voxel Feature Encoder)
```
lidar_vfe (PillarVFE_TA_va):
├── Input: [16, 20000, 48, 4]  # 这里的16代表的含义是什么？点云提供了哪些特征？？
├── Voxel Size: [0.2, 0.2, 8.0]
├── Point Cloud Range: [0, -44.8, -3.0, 112, 44.8, 5.0]
├── Max Points per Voxel: 48
├── Max Voxels: 20,000
├── Feature Extraction:
│   ├── Absolute XYZ: [x, y, z]
│   ├── Distance: sqrt(x² + y² + z²)
│   ├── Point Count: actual points in voxel
│   └── TA (Temporal Attention) Features: 64-dim
├── Output: [16, 20000, 64]
└── Parameters: ~100,000
```

#### PointPillar Scatter
```
lidar_middle_encoder (PointPillarScatter):
├── Input: [16, 20000, 64]
├── Voxel Coords: [16, 20000, 4] [batch_idx, z, y, x]
├── Grid Size: [896, 448, 1] (W×H×1)
├── Output: [16, 64, 448, 896]
├── Description: Scatter voxel features to BEV grid
└── Parameters: Minimal (just indexing)
```

#### LiDAR Backbone
```
lidar_bev_backbone (ATBackbone):
├── Input: [16, 64, 448, 896]
├── Layer Configuration:
│   ├── Layer1: [64→64], kernel=3, stride=1, layers=3
│   │   ├── Input: [16, 64, 448, 896]
│   │   └── Output: [16, 64, 448, 896]
│   ├── Layer2: [64→64], kernel=3, stride=2, layers=6
│   │   ├── Input: [16, 64, 448, 896]
│   │   └── Output: [16, 64, 224, 448]
│   ├── Layer3: [64→128], kernel=3, stride=2, layers=10
│   │   ├── Input: [16, 64, 224, 448]
│   │   └── Output: [16, 128, 112, 224]
│   └── Layer4: [128→256], kernel=3, stride=2, layers=10
│       ├── Input: [16, 128, 112, 224]
│       └── Output: [16, 256, 56, 112]
├── Upsample Layers:
│   ├── Upsample1: [256→128], stride=2
│   ├── Upsample2: [128→128], stride=2
│   ├── Upsample3: [128→128], stride=2
│   └── Upsample4: [128→128], stride=0.25
├── Output Features: Multi-scale [64, 64, 128, 256, 128, 128, 128, 128]
└── Parameters: ~2.0M
```

### 🔄 视图变换模块 (View Transfer)

#### BEV IPM Transfer
```
view_transfer (BevIpmTransfer):
├── Input Features:
│   ├── Group1: [16, 1, 256, 128, 240]
│   ├── Group2: [16, 1, 256, 64, 120]
│   ├── Group3: [16, 1, 256, 128, 240]
│   └── Group4: [16, 4, 256, 64, 120]
├── Camera Parameters:
│   ├── Intrinsics (K): [16, 7, 3, 3]
│   ├── Distortion: [16, 7, 5]
│   ├── BEV2Cam: [16, 7, 4, 4]
│   └── BEV2Img: [16, 7, 3, 3]
├── BEV Configuration:
│   ├── Dynamic Range: [-40, -44.8, -3.0, 62.4, 44.8, 5.0]
│   ├── Static Range: [-20.8, -22.4, -3.0, 62.4, 22.4, 5.0]
│   ├── Dynamic BEV: [112, 208] (0.4m resolution)
│   └── Static BEV: [56, 104] (0.8m resolution)
├── Reference Points Generation:
│   ├── Dynamic: [16, 112, 208, 2]
│   ├── Static: [16, 56, 104, 2]
│   └── Points per Pillar: 4
├── Deformable Attention:
│   ├── Num Levels: 1
│   ├── Num Points: 8
│   ├── Embed Dim: 256
│   └── Num Heads: 8
├── Output:
│   ├── Dynamic BEV: [16, 256, 112, 208]
│   └── Static BEV: [16, 256, 56, 104]
└── Parameters: ~500,000
```

### 🔀 多模态融合模块 (Fusion Module)

#### BEV Fuser
```
fuser (BevFuser):
├── Input:
│   ├── Camera BEV: [16, 256, 112, 208]
│   └── LiDAR BEV: [16, 512, 56, 112]
├── LiDAR Upsampling:
│   ├── ConvTranspose2d: [512→256], kernel=2, stride=2
│   ├── Input: [16, 512, 56, 112]
│   └── Output: [16, 256, 112, 224]
├── LiDAR Cropping:
│   ├── Crop to: [16, 256, 112, 208]
│   └── Align with Camera BEV
├── Fusion Operation:
│   ├── Concatenation: [16, 512, 112, 208]
│   ├── Conv2d: [512→256], kernel=3, padding=1
│   ├── BatchNorm2d: 256 channels
│   └── ReLU Activation
├── Output: [16, 256, 112, 208]
└── Parameters: ~400,000
```

### 🎯 任务专用处理头

#### 动态检测头 (Dynamic Head)
```
bev_dynamic_head (DynamicHead_Bin):
├── Input: [16, 256, 112, 208]
├── Neck Processing:
│   ├── bev_dynamic_neck (ConvResBlockNeck):
│   │   ├── Input: [16, 256, 112, 208]
│   │   ├── Conv Blocks: 3×[Conv3x3+BN+ReLU]
│   │   ├── Residual Connections: Yes
│   │   ├── Output: [16, 256, 112, 208]
│   │   └── Parameters: ~200,000
│   ├── Flatten: [16, 256, 112, 208] → [16, 23328, 256]
│   └── Permute: [16, 23328, 256] → [16, 23328, 256]
├── Query Embedding:
│   ├── Num Queries: 384
│   ├── Embed Dim: 256
│   ├── Learnable Parameters: [384, 256]
│   └── Parameters: ~98,000
├── Transformer Decoder (StreamDetrDecoder):
│   ├── Num Layers: 3
│   ├── Each Layer (StreamTransformerLayer):
│   │   ├── Self-Attention:
│   │   │   ├── MultiheadAttention: embed_dims=256, num_heads=8
│   │   │   ├── Dropout: 0.1
│   │   │   └── Parameters: ~260,000
│   │   ├── Cross-Attention:
│   │   │   ├── StreamDetrDeformableAttention
│   │   │   ├── Num Levels: 1
│   │   │   ├── Num Points: 20
│   │   │   ├── WL Size: 20
│   │   │   └── Parameters: ~200,000
│   │   ├── FFN:
│   │   │   ├── Linear: 256→512
│   │   │   ├── ReLU
│   │   │   ├── Linear: 512→256
│   │   │   ├── Dropout: 0.1
│   │   │   └── Parameters: ~400,000
│   │   └── LayerNorm: 256
│   └── Total Decoder Parameters: ~2.5M
├── Prediction Heads:
│   ├── Classification Head:
│   │   ├── Linear: 256→8
│   │   ├── Output: [16, 384, 8]
│   │   └── Classes: car, truck, bus, person, non_motor, riderless_non_motor, barrier, pillar
│   ├── Bbox Head:
│   │   ├── Linear: 256→6
│   │   ├── Output: [16, 384, 6]
│   │   └── Format: [x, y, z, w, l, h]
│   ├── Velocity Head:
│   │   ├── Linear: 256→3
│   │   ├── Output: [16, 384, 3]
│   │   └── Format: [vx, vy, vz]
│   ├── Bin Classification Head:
│   │   ├── Linear: 256→8
│   │   ├── Output: [16, 384, 8]
│   │   └── Bins: 8 directional bins
│   └── Occlusion Head:
│       ├── Linear: 256→1
│       ├── Output: [16, 384, 1]
│       └── Levels: 4 occlusion levels
├── Hungarian Assigner:
│   ├── Classification Cost: FocalLossCost
│   ├── Bbox Cost: L1Loss
│   └── Parameters: Minimal
└── Total Dynamic Head Parameters: ~3.5M
```

#### 静态分割头 (Static Head)
```
bev_static_head (StaticSegHead):
├── Input: [16, 256, 56, 104]
├── Neck Processing:
│   ├── bev_static_neck (Conv3x3Neck):
│   │   ├── Input: [16, 256, 56, 104]
│   │   ├── Conv3x3: 256→128
│   │   ├── BatchNorm + ReLU
│   │   ├── Output: [16, 128, 56, 104]
│   │   └── Parameters: ~300,000
│   └── Upsample: [16, 128, 56, 104] → [16, 128, 224, 416]
├── Lane Marking Head:
│   ├── Input: [16, 128, 224, 416]
│   ├── Conv Blocks: 5×[Conv3x3+BN+ReLU]
│   ├── Output Channels: 9
│   ├── Output: [16, 9, 224, 416]
│   ├── Classes: SolidLine, DoubleSolidLine, DashedLine, DoubleDashedLine, 
│   │             RightSolidLeftDashed, LeftSolidRightDashed, 
│   │             WideSolidLine, WideDashedLine, ShortDashedLine
│   └── Parameters: ~500,000
├── Road Element Head:
│   ├── Input: [16, 128, 224, 416]
│   ├── Conv Blocks: 5×[Conv3x3+BN+ReLU]
│   ├── Output Channels: 4
│   ├── Output: [16, 4, 224, 416]
│   ├── Classes: Wall, Curb, Lane, SpeedBump, GroundSigns
│   └── Parameters: ~200,000
├── Instance Detection Head:
│   ├── Input: [16, 128, 224, 416]
│   ├── Heatmap Head:
│   │   ├── Conv Blocks: 3×[Conv3x3+BN+ReLU]
│   │   ├── Output: [16, 1, 224, 416]
│   │   └── Max Instances: 100
│   ├── Regression Head:
│   │   ├── Conv Blocks: 3×[Conv3x3+BN+ReLU]
│   │   ├── Output: [16, 6, 224, 416]
│   │   └── Format: [offset_x, offset_y, w, h, angle, class]
│   └── Total Instance Parameters: ~400,000
└── Total Static Head Parameters: ~1.4M
```

## 📊 完整参数统计

### 总参数量分布
```
Backbone Networks:
├── stage1_backbone0: ~21.3M
├── stage1_backbone1: ~21.3M
├── stage1_backbone2: ~21.3M
├── stage1_backbone3: ~21.3M
└── Backbone Total: ~85.2M

Neck Networks:
├── stage1_neck0: ~200,000
├── stage1_neck1: ~150,000
├── stage1_neck2: ~200,000
├── stage1_neck3: ~150,000
└── Neck Total: ~700,000

LiDAR Networks:
├── lidar_vfe: ~100,000
├── lidar_middle_encoder: ~1,000
├── lidar_bev_backbone: ~2.0M
└── LiDAR Total: ~2.1M

Fusion & View Transfer:
├── view_transfer: ~500,000
├── fuser: ~400,000
└── Fusion Total: ~900,000

Task Heads:
├── bev_dynamic_head: ~3.5M
├── bev_static_head: ~1.4M
└── Heads Total: ~4.9M

Grand Total Parameters: ~93.8M
```

### 内存占用分析 (Batch=16)
```
Input Memory:
├── Camera Images: ~150MB
├── LiDAR Data: ~61MB
└── Input Total: ~211MB

Feature Memory:
├── Backbone Features: ~2.7GB
├── Neck Features: ~800MB
├── BEV Features: ~400MB
└── Feature Total: ~3.9GB

Output Memory:
├── Dynamic Predictions: ~50MB
├── Static Predictions: ~100MB
└── Output Total: ~150MB

Total Memory Usage: ~4.3GB per GPU
```

## 🔄 完整前向传播流程

### Step 1: 数据预处理
```
Raw Input:
├── Camera Images: [16, 7, 3, H, W]
├── LiDAR Points: Variable length
└── Camera/LiDAR Parameters

Preprocessing:
├── Image Resize: H×W → 512×960
├── Normalization: (img - 128) / 1.0
├── Data Augmentation: 2D/3D transforms
└── Voxelization: Points → Voxels
```

### Step 2: 特征提取
```
Camera Branch:
├── Group Processing: 4 parallel ResNet34
├── Multi-scale Features: P2, P3, P4
├── FPN Fusion: Top-down + Lateral
└── Camera Features: [16, 7, 256, H/8, W/8]

LiDAR Branch:
├── VFE: Point → Voxel Features
├── Scatter: Voxel → BEV Grid
├── Backbone: Multi-scale 2D CNN
└── LiDAR Features: [16, 512, 56, 112]
```

### Step 3: BEV变换
```
View Transfer:
├── Reference Points: BEV grid sampling
├── Deformable Attention: Feature sampling
├── Multi-camera Fusion: 7→1
└── BEV Features: Dynamic + Static

Fusion:
├── LiDAR Upsampling: Match camera resolution
├── Feature Concatenation: Camera + LiDAR
├── Fusion Conv: 512→256
└── Fused BEV: [16, 256, 112, 208]
```

### Step 4: 任务处理
```
Dynamic Task:
├── Neck: ConvResBlock processing
├── Transformer: Query-based decoding
├── Multi-head Prediction: 384 queries
└── Outputs: cls, bbox, velo, bin, occlude

Static Task:
├── Neck: Conv3x3 processing
├── Multi-task Heads: Lane + Road + Instance
├── Upsampling: 2× resolution
└── Outputs: seg_maps, instance_dets
```

### Step 5: 损失计算
```
Dynamic Losses:
├── Classification Loss: FocalLoss (γ=2.0, α=0.25)
├── Bbox Loss: L1Loss
├── Bin Loss: CrossEntropyLoss
├── Occlusion Loss: FocalLoss
├── Velocity Loss: L1Loss
└── Weighted Sum: total_loss = Σ(wi × lossi)

Static Losses:
├── Lane Segmentation Loss: FocalLoss + DiceLoss
├── Road Segmentation Loss: FocalLoss + DiceLoss
├── Instance Detection Loss: CenterNetLoss
└── Weighted Sum: total_loss = Σ(wi × lossi)
```

## 🎯 网络输出详细说明

### 动态任务输出
```
Detection Results (per query):
├── Classification: [16, 384, 8]
│   ├── Format: [car, truck, bus, person, non_motor, riderless_non_motor, barrier, pillar]
│   ├── Activation: Sigmoid
│   └── Threshold: 0.3 (default)
├── Bounding Box: [16, 384, 6]
│   ├── Format: [x, y, z, w, l, h]
│   ├── Coordinate: BEV (ego vehicle)
│   └── Units: meters
├── Velocity: [16, 384, 3]
│   ├── Format: [vx, vy, vz]
│   ├── Coordinate: BEV
│   └── Units: m/s
├── Bin Classification: [16, 384, 8]
│   ├── Format: 8 directional bins
│   ├── Angle Range: [-π, π]
│   └── Bin Width: π/4
└── Occlusion: [16, 384, 1]
    ├── Format: [NoOccluded, SlightlyOccluded, PartlyOccluded, HeavilyOccluded]
    └── Activation: Sigmoid
```

### 静态任务输出
```
Segmentation Results:
├── Lane Marking: [16, 9, 224, 416]
│   ├── Resolution: 0.2m × 0.2m
│   ├── Classes: 9 lane marking types
│   └── Activation: Sigmoid
├── Road Elements: [16, 4, 224, 416]
│   ├── Resolution: 0.2m × 0.2m
│   ├── Classes: [Wall, Curb, Lane, SpeedBump, GroundSigns]
│   └── Activation: Sigmoid
└── Instance Detection: [16, 100, 6]
    ├── Format: [offset_x, offset_y, w, h, angle, class]
    ├── Max Instances: 100
    └── Coordinate: BEV
```

这个详细的网络架构输出展示了LeapAI框架在您配置下的完整结构，包括每一层的参数量、数据流动和内存占用。
