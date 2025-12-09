#!/usr/bin/env python3
"""
LeapAI框架学习 - 第2步：配置系统和入口机制实践

这个练习将帮助您：
1. 深入理解配置系统的工作原理
2. 学习如何修改和扩展配置
3. 掌握入口机制的详细流程
4. 实践配置文件的编写和调试
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def step1_config_system_deep_dive():
    """第1步：配置系统深度解析"""
    
    print("=" * 60)
    print("⚙️  配置系统深度解析")
    print("=" * 60)
    
    print("\n📋 配置系统核心组件：")
    config_components = {
        "Config类": "leapai/utils/config.py - 配置文件加载和解析",
        "环境变量": "LEAPAI_TASK_CONFIG - 主配置文件路径",
        "分层配置": "主配置 + 子任务配置的层次结构",
        "动态加载": "支持Python和YAML格式的配置文件"
    }
    
    for component, description in config_components.items():
        print(f"  • {component}: {description}")
    
    # 演示配置加载过程
    print("\n🔧 配置加载流程演示：")
    loading_steps = [
        "1. 设置环境变量 LEAPAI_TASK_CONFIG",
        "2. Config.fromfile() 读取配置文件",
        "3. Python exec() 执行配置文件内容",
        "4. 提取所有非私有变量到配置字典",
        "5. 返回Config对象供后续使用"
    ]
    
    for step in loading_steps:
        print(f"  {step}")
    
    return True

def step2_create_sample_config():
    """第2步：创建示例配置文件"""
    
    print("\n" + "=" * 60)
    print("📝 创建示例配置文件")
    print("=" * 60)
    
    # 创建一个简化的配置示例
    sample_config_content = '''#!/usr/bin/env python3
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
'''
    
    # 写入配置文件
    config_path = Path("helloworld/demo_config.py")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(sample_config_content)
    
    print(f"✅ 已创建示例配置文件: {config_path}")
    print("\n📄 配置文件包含的主要部分：")
    config_sections = [
        "基础配置 (job_name, max_steps等)",
        "学习率配置 (float_lr, lr_scheduler等)",
        "多任务配置 (multi_task_config)",
        "数据配置 (batch_sizes, num_workers等)",
        "模型配置 (enable_lidar, bev_hw等)",
        "训练配置 (save_ckpt_interval, log_every_n_steps等)"
    ]
    
    for section in config_sections:
        print(f"  • {section}")
    
    return True

def step3_entry_mechanism_analysis():
    """第3步：入口机制详细分析"""
    
    print("\n" + "=" * 60)
    print("🚀 入口机制详细分析")
    print("=" * 60)
    
    print("\n📋 tools/main.py 核心流程：")
    
    # 模拟main.py的执行流程
    main_flow = '''
def main(args):
    """主函数执行流程"""
    
    # 1. 环境初始化
    seed_everything(args.seed)
    reset_gpu()
    
    # 2. 配置加载
    cfg = Config.fromfile(args.config)
    
    # 3. 预训练权重处理
    ckpt = args.ckpt if args.ckpt else cfg.get("float_pretrain", None)
    resume_ckpt = args.resume if args.resume else cfg.get("resume_ckpt", None)
    
    # 4. 组件构建
    with RegistryContext():
        runner = build_from_registry(cfg.runner)      # Lightning Trainer
        model = build_from_registry(cfg.graph_model)  # NodeGraph Model
        data_module = build_from_registry(cfg.data_module)  # DataModule
        
        # 5. 权重加载
        if ckpt:
            model = load_checkpoint(model, ckpt)
        
        # 6. 执行训练/验证/测试
        if args.state == "train":
            runner.fit(model=model, datamodule=data_module)
        elif args.state == "val":
            runner.validate(model, data_module)
        elif args.state == "test":
            runner.test(model, data_module)
'''
    
    print(main_flow)
    
    print("\n🔧 命令行参数解析：")
    cmd_args = [
        "--config: 主配置文件路径 (必需)",
        "--state: 运行模式 train/val/test/predict (必需)",
        "--with-val: 训练时是否验证 (可选)",
        "--ckpt: 预训练权重路径 (可选)",
        "--resume: 恢复训练的检查点路径 (可选)",
        "--seed: 随机种子 (默认0)",
        "--num-threads: CPU线程数 (默认12)"
    ]
    
    for arg in cmd_args:
        print(f"  • {arg}")
    
    return True

def step4_practice_config_modification():
    """第4步：配置修改实践"""
    
    print("\n" + "=" * 60)
    print("🛠️  配置修改实践")
    print("=" * 60)
    
    print("\n📝 常见配置修改场景：")
    
    modification_examples = {
        "调整学习率": "修改 float_lr 和 finetune_lr",
        "改变batch size": "修改 batch_sizes 字典",
        "调整训练步数": "修改 max_steps, train_steps, finetune_steps",
        "启用/禁用模块": "修改 enable_lidar, enable_dynamic_temporal 等",
        "调整日志频率": "修改 log_every_n_steps",
        "改变保存间隔": "修改 save_ckpt_interval"
    }
    
    for scenario, instruction in modification_examples.items():
        print(f"  • {scenario}: {instruction}")
    
    # 创建配置修改示例
    print("\n🔧 配置修改示例代码：")
    modification_code = '''
# 示例1：调整学习率
original_lr = 2e-4
new_lr = original_lr * 0.5  # 学习率减半

# 示例2：根据GPU数量调整batch size
num_gpus = 4
base_batch_size = 16
adjusted_batch_size = base_batch_size * num_gpus

# 示例3：动态配置加载
def get_config_by_mode(mode="debug"):
    if mode == "debug":
        return {
            "max_steps": 100,
            "batch_sizes": {"dynamic": {"train": 2, "val": 1}},
            "log_every_n_steps": 10
        }
    else:
        return {
            "max_steps": 10000,
            "batch_sizes": {"dynamic": {"train": 16, "val": 1}},
            "log_every_n_steps": 50
        }
'''
    
    print(modification_code)
    
    return True

def step5_debug_config_loading():
    """第5步：配置加载调试"""
    
    print("\n" + "=" * 60)
    print("🐛 配置加载调试")
    print("=" * 60)
    
    print("\n🔍 常见配置问题及解决方案：")
    
    common_issues = [
        {
            "问题": "配置文件路径错误",
            "原因": "文件不存在或路径不正确",
            "解决": "检查文件路径，使用绝对路径或正确的相对路径"
        },
        {
            "问题": "配置变量未定义",
            "原因": "配置文件中缺少必需的变量",
            "解决": "参考完整配置文件，确保所有必需变量都已定义"
        },
        {
            "问题": "环境变量未设置",
            "原因": "LEAPAI_TASK_CONFIG 环境变量未设置",
            "解决": "export LEAPAI_TASK_CONFIG=path/to/config.py"
        },
        {
            "问题": "Python语法错误",
            "原因": "配置文件中存在语法错误",
            "解决": "使用 python -m py_compile config.py 检查语法"
        }
    ]
    
    for issue in common_issues:
        print(f"\n❌ 问题: {issue['问题']}")
        print(f"   原因: {issue['原因']}")
        print(f"   解决: {issue['解决']}")
    
    # 创建调试脚本
    debug_script_content = '''#!/usr/bin/env python3
"""
配置加载调试脚本
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from leapai.utils.config import Config

def debug_config_loading():
    """调试配置加载过程"""
    
    print("🔍 开始调试配置加载...")
    
    # 1. 检查环境变量
    config_path = os.environ.get("LEAPAI_TASK_CONFIG")
    print(f"环境变量 LEAPAI_TASK_CONFIG: {config_path}")
    
    if not config_path:
        print("❌ 环境变量未设置")
        return False
    
    # 2. 检查文件存在性
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 3. 尝试加载配置
    try:
        cfg = Config.fromfile(config_path)
        print(f"✅ 配置加载成功")
        print(f"   配置项数量: {len(cfg)}")
        
        # 4. 检查关键配置项
        key_configs = ["job_name", "max_steps", "multi_task_config"]
        for key in key_configs:
            if hasattr(cfg, key):
                print(f"   {key}: {getattr(cfg, key)}")
            else:
                print(f"   ⚠️  缺少配置项: {key}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

if __name__ == "__main__":
    debug_config_loading()
'''
    
    debug_script_path = Path("helloworld/debug_config.py")
    with open(debug_script_path, 'w', encoding='utf-8') as f:
        f.write(debug_script_content)
    
    print(f"\n🛠️  已创建调试脚本: {debug_script_path}")
    print("使用方法: python helloworld/debug_config.py")
    
    return True

def interactive_exercise():
    """互动练习"""
    
    print("\n" + "=" * 60)
    print("🎯 互动练习")
    print("=" * 60)
    
    print("\n📝 练习任务：")
    exercises = [
        "1. 修改示例配置文件中的学习率为1e-4",
        "2. 将动态任务的batch size改为4",
        "3. 添加一个新的配置项 'experiment_name'",
        "4. 创建一个debug模式的配置变体"
    ]
    
    for exercise in exercises:
        print(f"  {exercise}")
    
    print("\n💡 提示：")
    tips = [
        "使用文本编辑器打开 helloworld/demo_config.py",
        "修改相应的配置项",
        "保存文件后使用调试脚本验证",
        "观察配置加载是否成功"
    ]
    
    for tip in tips:
        print(f"  • {tip}")
    
    return True

def main():
    """主函数"""
    
    print("🎓 LeapAI框架学习 - 第2阶段：配置系统和入口机制")
    print("本阶段将深入理解配置系统和入口机制的工作原理")
    
    try:
        # 执行学习步骤
        step1_config_system_deep_dive()
        step2_create_sample_config()
        step3_entry_mechanism_analysis()
        step4_practice_config_modification()
        step5_debug_config_loading()
        interactive_exercise()
        
        print("\n" + "=" * 60)
        print("🎉 第2阶段学习完成！")
        print("=" * 60)
        
        print("\n📋 下一步学习建议：")
        next_steps = [
            "1. 实践修改配置文件并验证",
            "2. 深入分析 tools/main.py 的源码",
            "3. 学习 leapai/registry.py 的实现细节",
            "4. 尝试运行一个简单的训练任务"
        ]
        
        for step in next_steps:
            print(f"  {step}")
        
        print("\n💡 记住：配置是框架的灵魂，掌握配置系统是高效开发的关键！")
        
    except Exception as e:
        print(f"❌ 学习过程中出现错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
