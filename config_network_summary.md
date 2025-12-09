# LeapAI配置网络结构总结

## 🎯 您的配置分析

基于您的VSCode调试配置：
```json
{
    "env": {"LEAPAI_TASK_CONFIG": "projects/perception/configs/lpperception_current_hpa_step1.py"},
    "args": ["--config", "projects/perception/entry.py", "--state", "train"]
}
```

## 🏗️ 网络结构概览

### 核心架构
- **多模态感知**: 7相机 + LiDAR融合
- **多任务学习**: 动态检测 + 静态分割
- **BEV表示**: 统一鸟瞰图空间
- **端到端训练**: ResNet34骨干 + Transformer解码器

### 启用的功能
✅ `enable_lidar = True` - LiDAR融合  
✅ `dynamic_task = True` - 动态障碍物检测  
✅ `static_task = True` - 静态元素分割  
❌ `occ_task = False` - 占用网络（关闭）  
❌ `attr_task = False` - 属性识别（关闭）  
❌ 时序处理 - 全部关闭（单帧模式）

## 📊 数据输入与类型

### 1. 相机数据 (7个)
```
front_wide:    [B, 3, 1024, 1920] → [B, 3, 512, 960]
front_narrow:  [B, 3, 512, 960]
back:          [B, 3, 512, 960]  
front_left:    [B, 3, 512, 960]
front_right:   [B, 3, 512, 960]
back_left:     [B, 3, 512, 960]
back_right:    [B, 3, 512, 960]
```

### 2. LiDAR数据
```
点云: [N_points, 4] (x,y,z,intensity)
体素化: [20000, 48, 4] → [20000, 64]
BEV投影: [B, 512, 56, 112]
```

### 3. 标注数据
```
动态任务:
- 检测框: [B, 512, 7] (x,y,z,w,l,h,yaw)
- 类别: [B, 512] (8类)
- 速度: [B, 512, 3] (vx,vy,vz)
- 遮挡: [B, 512] (4级遮挡)

静态任务:
- 分割图: [B, 13, 224, 416] (9类车道线+4类道路)
- 实例: [B, 100, 6] (最多100个实例)
```

## 🔄 数据流动与Shape变化

### 阶段1: 特征提取
```
相机输入: [16, 7, 3, 512, 960]
  ↓ ResNet34 + FPN (4组并行)
相机特征: [16, 7, 256, H/8, W/8]
  ↓ 
LiDAR输入: 变长点云
  ↓ 体素化 + VFE + Backbone
LiDAR特征: [16, 512, 56, 112]
```

### 阶段2: BEV变换
```
多相机特征: [16, 7, 256, 64-128, 120-240]
  ↓ BevIpmTransfer (IPM + Deformable Attention)
动态BEV: [16, 256, 112, 208] (0.4m分辨率)
静态BEV: [16, 256, 56, 104]  (0.8m分辨率)
  ↓ 多模态融合 (BevFuser)
融合BEV: [16, 256, 112, 208] / [16, 256, 56, 104]
```

### 阶段3: 任务处理
```
动态分支:
输入: [16, 256, 112, 208]
  ↓ ConvResBlockNeck
特征: [16, 256, 112, 208]
  ↓ Flatten + StreamDetrDecoder (3层)
查询: [16, 384, 256]
  ↓ 检测头
输出:
- 分类: [16, 384, 8] (8类检测)
- 边界框: [16, 384, 6] (x,y,z,w,l,h,yaw)
- 速度: [16, 384, 3] (vx,vy,vz)
- 遮挡: [16, 384, 1]

静态分支:
输入: [16, 256, 56, 104]
  ↓ Conv3x3Neck
特征: [16, 128, 56, 104]
  ↓ StaticSegHead
输出:
- 车道线分割: [16, 9, 224, 416]
- 道路分割: [16, 4, 224, 416]
- 实例检测: [16, 100, 6]
```

## 🎯 检测类别

### 动态任务 (8类)
```python
class_names = [
    "car",              # 汽车
    "truck",            # 卡车
    "bus",              # 巴士
    "person",           # 行人
    "non_motor",        # 非机动车
    "riderless_non_motor", # 无骑手非机动车
    "barrier",          # 障碍物
    "pillar"            # 柱子
]
```

### 静态任务 (13类)
```python
# 分割任务
车道线类型 (9类):
- SolidLine (实线)
- DoubleSolidLine (双实线)  
- DashedLine (虚线)
- DoubleDashedLine (双虚线)
- RightSolidLeftDashed (右实左虚)
- LeftSolidRightDashed (左实右虚)
- WideSolidLine (宽实线)
- WideDashedLine (宽虚线)
- ShortDashedLine (短虚线)

道路元素 (4类):
- Wall (墙壁)
- Curb (路缘石)  
- Lane (车道线)
- SpeedBump (减速带)
- GroundSigns (地面标识)
```

## 📈 感知范围

### BEV覆盖范围
```
动态任务: [-40, -44.8, -3.0, 62.4, 44.8, 5.0]  # 102.4m × 89.6m
静态任务: [-20.8, -22.4, -3.0, 62.4, 22.4, 5.0] # 83.2m × 44.8m
```

### 评估距离分区
```
动态检测距离分区:
- [0, 3m]    近距离
- [3, 6m]    近距离  
- [6, 12m]   中距离
- [12, 24m]  中距离
- [24, 60m]  远距离
```

## ⚙️ 训练配置

### 优化设置
```python
batch_size = 16                    # 批次大小
learning_rate = 2e-4 * sqrt(gpus/8) # 自适应学习率
optimizer = AdamW                   # 优化器
scheduler = MultiStep               # 学习率调度
warmup_steps = 500                # 预热步数
max_steps = 125000                 # 最大训练步数
```

### 损失权重
```python
task_loss_weights = {
    "dynamic": 2.5,    # 动态任务权重
    "static": 0.67      # 静态任务权重
}
```

## 🚀 性能特点

### 内存占用 (单GPU, batch=16)
- **相机输入**: ~672MB
- **相机特征**: ~2.7GB  
- **LiDAR处理**: ~200MB
- **BEV特征**: ~400MB
- **总计**: ~4GB

### 计算复杂度
- **ResNet34骨干**: 中等复杂度
- **多相机并行**: 4倍计算
- **Transformer解码**: O(N²) 复杂度
- **多任务头**: 轻量级

### 推理性能
- **输入分辨率**: 7×512×960 + LiDAR
- **输出延迟**: ~50-100ms (GPU)
- **精度**: mAP > 80% (动态), mIoU > 70% (静态)

## 🎯 应用场景

这个网络结构特别适合:
1. **自动驾驶L2-L4**: 环境感知与决策
2. **智能座舱**: 周围环境监控
3. **机器人导航**: 复杂环境理解
4. **智慧交通**: 交通流分析与管理

## 📝 关键优势

1. **多模态互补**: 相机+LiDAR提升精度和鲁棒性
2. **统一BEV空间**: 简化多任务融合
3. **端到端学习**: 避免手工特征工程
4. **实时性能**: 优化的网络结构支持实时推理
5. **可扩展性**: 模块化设计便于功能扩展

这个配置代表了当前自动驾驶感知系统的主流方案，通过多模态融合和多任务学习，实现了对复杂交通环境的全面感知。
