# LeapAI框架实践训练指南

## 🎯 概述

本指南将帮助您使用LeapAI框架运行一个完整的感知模型训练任务。我们将基于现有的配置文件，从环境准备到模型训练的完整流程。

## 📋 前置条件

### 1. 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+
- 足够的GPU内存（建议16GB+）

### 2. 数据准备
确保以下数据路径存在：
- 动态数据集：`/dahuafs/groupdata/Cameraalgorithm/hpa_perception/BEV_Dynamic_target/251001`
- 静态数据集：`/dahuafs/groupdata/Cameraalgorithm/bev_perception/BEV_Static_map/train_v2.0/v2.8/8650/earlyfusion_v1`
- 预训练模型：`/dahuafs/groupdata/share/perception/release/v4.11/torch/v4.11.ckpt`

## 🚀 实践步骤

### 步骤1：环境配置

#### 1.1 设置环境变量
```bash
export LEAPAI_TASK_CONFIG="projects/perception/configs/lpperception_current_hpa_step1.py"
export RCNUM=1  # 节点数量
export GPU_NUM=1  # 每节点GPU数量
export my_debug="yes"  # 调试模式，使用小数据集
```

#### 1.2 验证配置文件
```python
# 验证配置加载
from leapai.utils.config import Config
import os

config_path = os.environ["LEAPAI_TASK_CONFIG"]
config = Config.fromfile(config_path)
print(f"任务名称: {config.job_name}")
print(f"启用LiDAR: {config.enable_lidar}")
print(f"动态任务: {config.dynamic_task}")
print(f"静态任务: {config.static_task}")
```

### 步骤2：数据验证

#### 2.1 检查数据集路径
```python
import os

def check_data_paths(config):
    """检查数据集路径是否存在"""
    missing_paths = []
    
    # 检查训练数据
    if config.dynamic_task:
        dynamic_path = config.dynamic_train_set_dir
        if not os.path.exists(dynamic_path):
            missing_paths.append(f"动态训练数据: {dynamic_path}")
    
    if config.static_task:
        static_path = config.static_train_set_dir
        if not os.path.exists(static_path):
            missing_paths.append(f"静态训练数据: {static_path}")
    
    # 检查预训练模型
    if config.float_pretrain and not os.path.exists(config.float_pretrain):
        missing_paths.append(f"预训练模型: {config.float_pretrain}")
    
    return missing_paths

missing = check_data_paths(config)
if missing:
    print("缺少以下路径:")
    for path in missing:
        print(f"  - {path}")
else:
    print("所有数据路径检查通过!")
```

#### 2.2 验证数据列表文件
```python
def validate_data_lists(config):
    """验证数据列表文件"""
    for task_name, task_config in config.train_set_info_path.items():
        if "online" in task_config:
            for data_list in task_config["online"]:
                if os.path.exists(data_list):
                    with open(data_list, 'r') as f:
                        lines = f.readlines()
                    print(f"{task_name} - {os.path.basename(data_list)}: {len(lines)} 个样本")
                else:
                    print(f"警告: {data_list} 不存在")

validate_data_lists(config)
```

### 步骤3：模型配置验证

#### 3.1 检查模型组件
```python
def validate_model_config():
    """验证模型配置"""
    from projects.perception import model_base
    
    print("基础模型节点:")
    for node_name in model_base.base_nodes.keys():
        print(f"  - {node_name}")
    
    print(f"\n相机配置: {len(model_base.camera_names)} 个相机")
    print(f"BEV尺寸: {model_base.bev_hw}")
    print(f"LiDAR范围: {model_base.lidar_range}")

validate_model_config()
```

#### 3.2 验证任务配置
```python
def validate_task_configs():
    """验证任务配置"""
    from projects.perception import dynamic, static
    
    print("动态任务配置:")
    print(f"  类别数量: {dynamic.num_classes}")
    print(f"  类别: {dynamic.class_names}")
    print(f"  最大对象数: {dynamic.max_objects}")
    
    print("\n静态任务配置:")
    print(f"  标签类型: {static.label_names}")
    print(f"  损失权重: {static.loss_weights}")

validate_task_configs()
```

### 步骤4：训练启动

#### 4.1 创建训练脚本
```python
# run_training.py
import os
import sys
sys.path.append('/dahuafs/userdata/40359/Leapnet_master')

def main():
    """主训练函数"""
    # 设置环境变量
    os.environ["LEAPAI_TASK_CONFIG"] = "projects/perception/configs/lpperception_current_hpa_step1.py"
    os.environ["RCNUM"] = "1"
    os.environ["GPU_NUM"] = "1"
    os.environ["my_debug"] = "yes"  # 调试模式
    
    # 导入并运行
    from projects.perception.entry import runner
    
    print("开始训练...")
    print(f"配置文件: {os.environ['LEAPAI_TASK_CONFIG']}")
    print(f"GPU数量: {os.environ['GPU_NUM']}")
    print(f"调试模式: {os.environ['my_debug']}")
    
    # 创建trainer
    trainer_config = runner
    print(f"训练器配置: {trainer_config}")
    
    # 这里可以进一步配置和启动训练
    print("训练配置验证完成!")

if __name__ == "__main__":
    main()
```

#### 4.2 运行训练
```bash
# 方式1: 直接运行Python脚本
python run_training.py

# 方式2: 使用框架入口
python -m projects.perception.entry

# 方式3: 使用torch.distributed（多GPU）
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    projects/perception/entry.py
```

### 步骤5：监控和调试

#### 5.1 训练监控
```python
def monitor_training():
    """监控训练过程"""
    import torch
    import time
    
    # 检查GPU使用情况
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # 检查内存使用
    import psutil
    memory = psutil.virtual_memory()
    print(f"系统内存使用: {memory.percent}% ({memory.used / 1e9:.1f} GB / {memory.total / 1e9:.1f} GB)")

monitor_training()
```

#### 5.2 调试工具
```python
def debug_training_step():
    """调试训练步骤"""
    import torch
    from leapai.utils.config import Config
    from projects.perception.entry import MAIN_CFG, TASK_CFGS
    
    print("=== 调试信息 ===")
    print(f"主配置任务数量: {len(MAIN_CFG.multi_task_config)}")
    
    for task_name, task_config in TASK_CFGS.items():
        print(f"\n任务: {task_name}")
        print(f"  节点数量: {len(task_config.nodes) if hasattr(task_config, 'nodes') else 'N/A'}")
        print(f"  数据集配置: {'✓' if hasattr(task_config, 'get_train_dataset') else '✗'}")
        print(f"  拓扑配置: {'✓' if hasattr(task_config, 'node_topology') else '✗'}")

debug_training_step()
```

### 步骤6：常见问题解决

#### 6.1 数据路径问题
```python
def fix_data_paths():
    """修复常见的数据路径问题"""
    import os
    
    # 检查并创建必要的目录
    required_dirs = [
        "./logs",
        "./checkpoints", 
        "./visualization",
        "./data"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

fix_data_paths()
```

#### 6.2 内存优化
```python
def optimize_memory_usage():
    """优化内存使用"""
    import torch
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU缓存已清理")
    
    # 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
    print("内存分配策略已设置")

optimize_memory_usage()
```

#### 6.3 配置验证
```python
def validate_training_config():
    """验证训练配置的完整性"""
    from leapai.utils.config import Config
    import os
    
    config_path = os.environ.get("LEAPAI_TASK_CONFIG")
    if not config_path:
        raise ValueError("未设置LEAPAI_TASK_CONFIG环境变量")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = Config.fromfile(config_path)
    
    # 验证必要的配置项
    required_keys = [
        'job_name', 'batch_size', 'max_steps', 'float_lr',
        'multi_task_config', 'train_set_info_path', 'val_set_info_path'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"配置文件缺少必要的键: {missing_keys}")
    
    print("配置验证通过!")
    return config

try:
    config = validate_training_config()
except Exception as e:
    print(f"配置验证失败: {e}")
```

## 🎯 完整训练示例

### 示例1：调试模式训练
```python
#!/usr/bin/env python3
"""
调试模式训练示例
使用小数据集快速验证训练流程
"""

import os
import sys
sys.path.append('/dahuafs/userdata/40359/Leapnet_master')

def setup_debug_environment():
    """设置调试环境"""
    os.environ["LEAPAI_TASK_CONFIG"] = "projects/perception/configs/lpperception_current_hpa_step1.py"
    os.environ["RCNUM"] = "1"
    os.environ["GPU_NUM"] = "1"
    os.environ["my_debug"] = "yes"
    
    print("=== 调试环境设置 ===")
    print(f"配置文件: {os.environ['LEAPAI_TASK_CONFIG']}")
    print(f"节点数: {os.environ['RCNUM']}")
    print(f"GPU数: {os.environ['GPU_NUM']}")
    print(f"调试模式: {os.environ['my_debug']}")

def run_debug_training():
    """运行调试训练"""
    try:
        setup_debug_environment()
        
        # 验证配置
        config = validate_training_config()
        print(f"任务名称: {config.job_name}")
        
        # 验证数据路径
        missing_paths = check_data_paths(config)
        if missing_paths:
            print("警告: 发现缺失的数据路径")
            for path in missing_paths:
                print(f"  - {path}")
        
        # 监控系统资源
        monitor_training()
        
        print("\n=== 调试训练准备完成 ===")
        print("可以开始运行实际训练了!")
        
    except Exception as e:
        print(f"调试训练设置失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug_training()
```

### 示例2：完整训练流程
```python
#!/usr/bin/env python3
"""
完整训练流程示例
"""

import os
import sys
sys.path.append('/dahuafs/userdata/40359/Leapnet_master')

def setup_production_environment():
    """设置生产环境"""
    os.environ["LEAPAI_TASK_CONFIG"] = "projects/perception/configs/lpperception_current_hpa_step1.py"
    os.environ["RCNUM"] = "1"
    os.environ["GPU_NUM"] = "4"  # 使用4个GPU
    # 不设置my_debug，使用完整数据集
    
    print("=== 生产环境设置 ===")
    print(f"配置文件: {os.environ['LEAPAI_TASK_CONFIG']}")
    print(f"节点数: {os.environ['RCNUM']}")
    print(f"GPU数: {os.environ['GPU_NUM']}")

def run_production_training():
    """运行生产训练"""
    try:
        setup_production_environment()
        
        # 验证配置
        config = validate_training_config()
        print(f"任务名称: {config.job_name}")
        print(f"最大步数: {config.max_steps}")
        print(f"批次大小: {config.batch_size}")
        
        # 检查预训练模型
        if config.float_pretrain and os.path.exists(config.float_pretrain):
            print(f"预训练模型: {config.float_pretrain}")
        else:
            print("警告: 未找到预训练模型")
        
        print("\n=== 生产训练准备完成 ===")
        print("建议使用分布式训练命令启动:")
        print("python -m torch.distributed.launch --nproc_per_node=4 projects/perception/entry.py")
        
    except Exception as e:
        print(f"生产训练设置失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_production_training()
```

## 🔧 高级配置

### 多GPU训练
```bash
# 4GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12345 \
    projects/perception/entry.py

# 多节点训练 (节点1)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    projects/perception/entry.py

# 多节点训练 (节点2)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    projects/perception/entry.py
```

### 混合精度训练
```python
# 在配置文件中添加
precision = "16-mixed"  # 使用混合精度
use_backbone_amp = True  # Backbone使用AMP
```

### 梯度累积
```python
# 在配置文件中调整
accumulate_grad_batches = 2  # 梯度累积2步
batch_size = 8  # 减小批次大小
# 有效批次大小 = 8 * 2 = 16
```

## 📊 监控和日志

### TensorBoard监控
```python
# 启动TensorBoard
tensorboard --logdir=./logs --port=6006

# 在配置中启用TensorBoard
logger = dict(
    type="TensorBoardLogger",
    save_dir="./logs",
    name=config.job_name,
)
```

### 训练脚本监控
```python
def monitor_training_progress(log_file="./logs/training.log"):
    """监控训练进度"""
    import re
    from datetime import datetime
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 解析损失信息
    losses = []
    for line in lines[-100:]:  # 最近100行
        if "loss" in line.lower():
            # 使用正则表达式提取损失值
            match = re.search(r'loss:\s*([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))
    
    if losses:
        print(f"最近损失趋势: {losses[-10:]}")
        print(f"当前损失: {losses[-1]:.4f}")
        print(f"损失变化: {losses[-1] - losses[-10]:.4f}")

# 使用示例
monitor_training_progress()
```

## 🚨 故障排除

### 常见错误及解决方案

#### 1. CUDA内存不足
```python
# 解决方案1: 减小批次大小
batch_size = 4  # 从16减少到4

# 解决方案2: 启用梯度检查点
from torch.utils.checkpoint import checkpoint

# 解决方案3: 清理GPU缓存
torch.cuda.empty_cache()
```

#### 2. 数据加载错误
```python
# 检查数据路径
def debug_data_loading():
    """调试数据加载"""
    from projects.perception.dataset import LeapDataset
    
    # 创建一个简单的数据集实例
    sample_case = {
        "case_path": "/path/to/sample/case.json",
        "scene_name": "test_scene"
    }
    
    try:
        dataset = LeapDataset(
            case_info=sample_case,
            camera_names=["front_wide", "front_left", "front_right"],
            pipeline=[],
            task_name="dynamic"
        )
        print("数据集创建成功")
    except Exception as e:
        print(f"数据集创建失败: {e}")

debug_data_loading()
```

#### 3. 配置文件错误
```python
def debug_config_loading():
    """调试配置加载"""
    import os
    from leapai.utils.config import Config
    
    config_path = os.environ.get("LEAPAI_TASK_CONFIG")
    if not config_path:
        print("错误: 未设置LEAPAI_TASK_CONFIG")
        return
    
    try:
        config = Config.fromfile(config_path)
        print("配置加载成功")
        print(f"任务名称: {config.job_name}")
        
        # 检查关键配置
        required_keys = ['batch_size', 'max_steps', 'multi_task_config']
        for key in required_keys:
            if hasattr(config, key):
                print(f"✓ {key}: {getattr(config, key)}")
            else:
                print(f"✗ {key}: 缺失")
                
    except Exception as e:
        print(f"配置加载失败: {e}")

debug_config_loading()
```

## 📈 性能优化

### 1. 数据加载优化
```python
# 增加数据加载器工作进程
num_workers = {
    "dynamic": {"train": 8, "val": 4},
    "static": {"train": 8, "val": 4},
}

# 启用持久化工作进程
persistent_workers = True

# 使用pin_memory
pin_memory = True
```

### 2. 模型优化
```python
# 使用编译优化 (PyTorch 2.0+)
model = torch.compile(model)

# 使用更高效的backbone
# 例如：将ResNet34替换为EfficientNet
```

### 3. 训练策略优化
```python
# 学习率调度
lr_scheduler = "cosine"  # 余弦退火

# 预热策略
warmup_steps = 1000

# 梯度裁剪
gradient_clip_val = 35.0
```

## 🎯 总结

本实践指南涵盖了：

1. **环境配置**: 设置必要的环境变量和依赖
2. **数据验证**: 确保数据路径和格式正确
3. **模型配置**: 验证模型组件和任务配置
4. **训练启动**: 多种训练启动方式
5. **监控调试**: 实时监控和调试
6. **性能优化**: 提升训练效率的各种技巧
7. **故障排除**: 常见问题的解决方案

## 📚 下一步

完成实践训练后，您可以：

1. **尝试不同配置**: 修改超参数和模型架构
2. **添加新任务**: 基于现有框架添加新的感知任务
3. **优化性能**: 进一步优化训练和推理性能
4. **部署应用**: 将训练好的模型部署到实际应用中

## 🔗 相关资源

- [LeapAI学习指南](LeapAI_Learning_Guide.md)
- [快速开始指南](Quick_Start_Guide.md)
- [配置系统详解](config_loading_troubleshooting.md)
- [分布式训练指南](distributed_training_detailed_analysis.md)

---

**注意**: 本指南基于LeapAI框架的当前版本编写，某些配置可能需要根据您的具体环境进行调整。建议在实际使用前先在调试模式下验证配置的正确性。
