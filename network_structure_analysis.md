    "hue": 0,            # 色调变化
    "resize": (-0.06, 0.11),  # 缩放范围
    "crop": (-0.05, 0.05),     # 裁剪范围
    "rot": (-5.4, 5.4),        # 旋转范围
    "flip": True               # 翻转
}

# 3D增强
data_config_3d = {
    "rotate_z": [[1.0, (-22.5, 22.5)]],  # Z轴旋转
    "scale": (1.0, 1.0),                 # 缩放
    "x_trans": (-0, 0),                   # X轴平移
    "y_trans": (-2, 2),                   # Y轴平移
    "z_trans": (-0, 0)                    # Z轴平移
}
```

## 🚀 部署与推理

### 1. ONNX导出支持
```python
# ONNX配置
ENV_ONNX = eval(os.environ.get("ONNX", "False"))

if ENV_ONNX:
    view_transfer["type"] = BevIpmTransferOnnx
    view_transfer["undistort_as_sdk"] = True
    
    # ONNX输入处理
    def onnx_input_process(model, batch):
        # 生成参考点
        generate_reference_points(...)
        
        # 处理图像输入
        imgs = []
        for cam_name in camera_names:
            imgs.append(batch["image"][cam_name])
        
        # 处理LiDAR输入
        if enable_lidar:
            voxels = batch["voxels"]
            voxel_num_points = batch["voxel_num_points"]
            voxel_coords = batch["voxel_coords"]
        
        return onnx_batch
```

### 2. 推理优化
```python
# 精度配置
precision = "16-mixed"  # 混合精度推理

# 批处理优化
batch_size = 1  # 推理时通常使用batch_size=1

# 后处理优化
conf_thres = {
    "lane": 0.75,        # 车道线置信度阈值
    "road": 0.75,        # 道路置信度阈值
    "hm_det": 0.60,      # 检测热图阈值
    "person": 0.30,      # 行人检测阈值
    "car": 0.30          # 车辆检测阈值
}
```

## 📈 性能指标与评估

### 1. 动态任务评估
```python
# 评估范围配置
eval_range_total = [-40, 62.4, -44.8, 44.8]  # [x_min, x_max, y_min, y_max]
eval_range_list = [
    [0, 3],      # 近距离 0-3m
    [3, 6],      # 近距离 3-6m  
    [6, 12],     # 中距离 6-12m
    [12, 24],    # 中距离 12-24m
    [24, 60]     # 远距离 24-60m
]

# 类别分组
dist_names = {
    "MOD": ["car", "truck", "bus", "other"],      # 机动车
    "VRU": ["person", "non_motor"],               # 弱势道路使用者
    "SOD": ["riderless_non_motor", "barrier", "pillar"]  # 静态障碍物
}

# 距离阈值
distance_threshold = 1.5  # IoU距离阈值
```

### 2. 静态任务评估
```python
# 分割类别
label_names = {
    "Seg": ["Wall", "Curb", "Lane"],           # 分割任务
    "Det": ["SpeedBump", "GroundSigns"]        # 检测任务
}

# 距离阈值
dist_thres = {
    "Wall": 0.25,        # 墙壁
    "Curb": 0.25,       # 路缘石
    "Lane": 0.25,       # 车道线
    "SpeedBump": 0.25,  # 减速带
    "GroundSigns": 0.25  # 地面标识
}

# 系列评估
static_series_eval = True
series_eval_conf = [0.9, 0.85, 0.8, 0.75, 0.7]
series_dis_thresh = 0.25
```

## 🔧 调试与可视化

### 1. 可视化配置
```python
# BEV可视化
draw_static_cfg = {
    "lane_conf": 0.75,
    "road_conf": 0.75,
    "freespace_conf": 0.5,
    "stopline_conf": 0.6,
    "crosswalk_conf": 0.9,
    "valid_range": [-150, 20, -25, 25],
    "coords": lidar_range["static"],
    "label_h": 224,
    "label_w": 416
}

# 动态可视化
vis_class_names = [
    "car", "truck", "bus", "person",
    "non_motor", "riderless_non_motor", 
    "barrier", "pillar"
]

vis_color_dt = {
    "person": [0, 97, 255],
    "non_motor": [255, 255, 0],
    "car": [255, 255, 255],
    "truck": [240, 32, 160],
    "bus": [0, 255, 0],
    "riderless_non_motor": [128, 128, 128],
    "barrier": [128, 128, 128],
    "pillar": [128, 128, 128]
}
```

### 2. 调试工具
```python
# Debug模式
my_debug = os.environ.get("my_debug", None)
if my_debug == "yes":
    # 使用小数据集快速测试
    mini_dataset_length = {"dynamic": 8, "static": 8}
    batch_sizes = {
        "dynamic": {"train": 1, "val": 1},
        "static": {"train": 2, "val": 1}
    }

# 性能分析
profiler = "simple"  # 启用性能分析器

# 梯度检查
detect_anomaly = False  # 梯度异常检测
```

## 📊 总结

### 网络特点
1. **多模态融合**: 结合7相机+LiDAR的丰富感知信息
2. **多任务学习**: 同时处理动态检测和静态分割任务
3. **BEV表示**: 统一的鸟瞰图空间表示
4. **端到端训练**: 从原始数据到最终结果的端到端学习

### 性能优势
1. **高精度**: 多模态互补提升检测精度
2. **强鲁棒性**: 多视角冗余增强系统鲁棒性
3. **实时性**: 优化的网络结构支持实时推理
4. **可扩展性**: 模块化设计便于功能扩展

### 应用场景
1. **自动驾驶**: L2-L4级自动驾驶感知系统
2. **智能交通**: 交通监控与管理
3. **机器人导航**: 复杂环境感知与导航
4. **智慧城市**: 城市环境理解与分析

这个网络结构代表了当前自动驾驶感知领域的先进水平，通过多模态融合和多任务学习，实现了对复杂交通环境的全面感知理解。
