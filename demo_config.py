#!/usr/bin/env python3
"""
LeapAI框架学习示例配置文件
这是一个简化的感知任务配置示例
"""

# ===== 基础配置 =====
job_name = "leapai_learning_demo"
max_steps = 1000
train_steps = 800
finetune_steps = 200
warmup_steps = 100

# ===== 学习率配置 =====
float_lr = 2e-4
finetune_lr = 1e-4
lr_scheduler = "cosine"

# ===== 多任务配置 =====
multi_task_config = {
    "dynamic": "helloworld/dynamic_task_demo.py",
    "static": "helloworld/static_task_demo.py"
}

# ===== 数据配置 =====
batch_sizes = {
    "dynamic": {"train": 8, "val": 1},
    "static": {"train": 8, "val": 1}
}

num_workers = {
    "dynamic": {"train": 4, "val": 2},
    "static": {"train": 4, "val": 2}
}

# ===== 数据路径配置 =====
train_set_info_path = {
    "dynamic": {
        "online": ["path/to/dynamic_train_list.txt"]
    },
    "static": {
        "online": ["path/to/static_train_list.txt"]
    }
}

val_set_info_path = {
    "dynamic": {
        "test_dynamic": {
            "path": ["path/to/dynamic_test_list.txt"]
        }
    },
    "static": {
        "test_static": {
            "path": ["path/to/static_test_list.txt"]
        }
    }
}

# ===== 模型配置 =====
enable_lidar = False
enable_dynamic_temporal = False
enable_static_temporal = False

# ===== BEV配置 =====
bev_hw = {
    "dynamic": (112, 128),
    "static": (56, 104)
}

lidar_range = {
    "dynamic": [-40, -44.8, -3.0, 62.4, 44.8, 5.0],
    "static": [-20.8, -22.4, -3.0, 62.4, 22.4, 5.0]
}

# ===== 训练配置 =====
save_ckpt_interval = 200
log_every_n_steps = 50
accumulate_grad_batches = 1

# ===== 分布式配置 =====
devices_id = "auto"
precision = "32"

# ===== 其他配置 =====
eval_with_visualize = True
use_streaming = {
    "dynamic": False,
    "static": False
}
