#!/usr/bin/env python3
"""
LeapAI框架学习 - tools/main.py 详细实践

本脚本专门用于深入理解和实践 tools/main.py 的各个功能模块
包括参数解析、配置加载、组件构建和执行流程
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_command_line_arguments():
    """分析命令行参数"""
    
    print("=" * 60)
    print("🔧 命令行参数分析")
    print("=" * 60)
    
    try:
        # 导入main.py的parse_args函数
        from tools.main import parse_args
        
        print("✅ 成功导入 parse_args 函数")
        
        # 模拟命令行参数
        test_args = [
            "--config", "projects/perception/entry.py",
            "--state", "train",
            "--with-val",
            "--seed", "42",
            "--num-threads", "8"
        ]
        
        # 临时修改sys.argv来测试参数解析
        original_argv = sys.argv.copy()
        sys.argv = ["main.py"] + test_args
        
        try:
            args = parse_args()
            print("✅ 参数解析成功")
            
            print(f"\n📋 解析结果:")
            print(f"  • config: {args.config}")
            print(f"  • state: {args.state}")
            print(f"  • with_val: {args.with_val}")
            print(f"  • seed: {args.seed}")
            print(f"  • ckpt: {args.ckpt}")
            print(f"  • resume: {args.resume}")
            print(f"  • num_threads: {args.num_threads}")
            print(f"  • local_rank: {args.local_rank}")
            print(f"  • verbose: {args.verbose}")
            
        finally:
            sys.argv = original_argv
        
        return True
        
    except Exception as e:
        print(f"❌ 命令行参数分析失败: {e}")
        return False

def analyze_config_loading():
    """分析配置加载过程"""
    
    print("\n" + "=" * 60)
    print("📄 配置加载分析")
    print("=" * 60)
    
    try:
        from leapai.utils.config import Config
        
        print("✅ 成功导入 Config 类")
        
        # 设置必要的环境变量
        os.environ["LEAPAI_TASK_CONFIG"] = "projects/perception/configs/lpperception_current_hpa_step1.py"
        os.environ["RCNUM"] = "1"
        os.environ["GPU_NUM"] = "1"
        
        # 测试配置文件路径
        config_paths = [
            "projects/perception/configs/lpperception_current_hpa_step1.py",
            "projects/perception/entry.py"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                print(f"\n📁 分析配置文件: {config_path}")
                
                try:
                    if "entry.py" in config_path:
                        print("  ⚠️  entry.py 需要环境变量 LEAPAI_TASK_CONFIG")
                        print("  💡 跳过直接加载，分析依赖的配置文件")
                        continue
                    
                    cfg = Config.fromfile(config_path)
                    print(f"✅ 配置加载成功")
                    
                    # 分析配置结构
                    if hasattr(cfg, 'text'):
                        print(f"  • 配置文本长度: {len(cfg.text)} 字符")
                    
                    # 查找关键配置项
                    key_configs = ['runner', 'graph_model', 'data_module']
                    for key in key_configs:
                        if hasattr(cfg, key):
                            config_value = getattr(cfg, key)
                            if isinstance(config_value, dict):
                                print(f"  • {key}: {type(config_value).__name__} (包含 {len(config_value)} 个键)")
                                if 'type' in config_value:
                                    print(f"    - type: {config_value['type']}")
                            else:
                                print(f"  • {key}: {type(config_value).__name__}")
                        else:
                            print(f"  • {key}: 未找到")
                    
                except Exception as e:
                    print(f"❌ 配置加载失败: {e}")
            else:
                print(f"⚠️  配置文件不存在: {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载分析失败: {e}")
        return False

def analyze_registry_mechanism():
    """分析注册机制"""
    
    print("\n" + "=" * 60)
    print("🏗️ 注册机制分析")
    print("=" * 60)
    
    try:
        from leapai.registry import RegistryContext, build_from_registry
        
        print("✅ 成功导入注册相关函数")
        
        # 分析注册机制的工作原理
        print("\n📋 RegistryContext 功能:")
        print("  • 提供组件注册的上下文管理")
        print("  • 支持组件的动态加载和卸载")
        print("  • 避免全局命名空间污染")
        
        print("\n📋 build_from_registry 功能:")
        print("  • 根据配置动态构建组件")
        print("  • 支持参数传递和验证")
        print("  • 返回可用的组件实例")
        
        # 模拟组件构建过程
        print("\n🔧 模拟组件构建:")
        
        # 创建示例配置
        example_configs = {
            "runner": {
                "type": "pl_trainer",
                "max_epochs": 100,
                "gpus": 1
            },
            "graph_model": {
                "type": "perception_model",
                "backbone": "resnet50"
            },
            "data_module": {
                "type": "multitask_datamodule",
                "batch_size": 8
            }
        }
        
        for component_name, config in example_configs.items():
            print(f"  • {component_name}:")
            print(f"    - type: {config.get('type', 'Unknown')}")
            print(f"    - 参数数量: {len(config) - 1}")
        
        print("\n💡 注意: 实际的组件构建需要在 RegistryContext 上下文中进行")
        
        return True
        
    except Exception as e:
        print(f"❌ 注册机制分析失败: {e}")
        return False

def analyze_execution_modes():
    """分析执行模式"""
    
    print("\n" + "=" * 60)
    print("🎯 执行模式分析")
    print("=" * 60)
    
    try:
        # 分析不同的执行模式
        modes = {
            "train": {
                "description": "训练模式",
                "features": ["模型训练", "验证（可选）", "检查点保存", "日志记录"],
                "config_adjustments": ["num_sanity_val_steps=0", "limit_val_batches=0"]
            },
            "val": {
                "description": "验证模式", 
                "features": ["模型验证", "指标计算", "结果输出"],
                "config_adjustments": ["val_check_interval=None"]
            },
            "test": {
                "description": "测试模式",
                "features": ["模型测试", "最终评估", "性能指标"],
                "config_adjustments": []
            },
            "predict": {
                "description": "预测模式",
                "features": ["推理预测", "结果生成", "批量处理"],
                "config_adjustments": []
            }
        }
        
        print("📋 支持的执行模式:")
        for mode, info in modes.items():
            print(f"\n  🎯 {mode.upper()} - {info['description']}")
            print(f"    功能特性:")
            for feature in info['features']:
                print(f"      • {feature}")
            if info['config_adjustments']:
                print(f"    配置调整:")
                for adjustment in info['config_adjustments']:
                    print(f"      • {adjustment}")
        
        # 分析模式切换逻辑
        print(f"\n🔄 模式切换逻辑:")
        print("  1. 根据 --state 参数选择执行模式")
        print("  2. 根据模式调整配置参数")
        print("  3. 构建对应的组件")
        print("  4. 调用相应的执行方法")
        
        return True
        
    except Exception as e:
        print(f"❌ 执行模式分析失败: {e}")
        return False

def practice_command_examples():
    """实践命令示例"""
    
    print("\n" + "=" * 60)
    print("💻 命令示例实践")
    print("=" * 60)
    
    try:
        # 定义不同的使用场景
        scenarios = {
            "基础训练": {
                "command": "python tools/main.py --config projects/perception/entry.py --state train --with-val",
                "description": "标准的训练任务，包含验证"
            },
            "预训练微调": {
                "command": "python tools/main.py --config projects/perception/entry.py --state train --ckpt /path/to/pretrain.ckpt --with-val",
                "description": "从预训练权重开始微调"
            },
            "恢复训练": {
                "command": "python tools/main.py --config projects/perception/entry.py --state train --resume /path/to/checkpoint.ckpt --with-val",
                "description": "从检查点恢复训练"
            },
            "模型验证": {
                "command": "python tools/main.py --config projects/perception/entry.py --state val --ckpt /path/to/checkpoint.ckpt",
                "description": "验证训练好的模型"
            },
            "分布式训练": {
                "command": "python -m torch.distributed.launch --nproc_per_node=4 tools/main.py --config projects/perception/entry.py --state train --with-val",
                "description": "多GPU分布式训练"
            }
        }
        
        print("📋 常用命令示例:")
        for scenario, info in scenarios.items():
            print(f"\n  🎯 {scenario}:")
            print(f"    描述: {info['description']}")
            print(f"    命令: {info['command']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 命令示例实践失败: {e}")
        return False

def analyze_main_workflow():
    """分析main.py工作流程"""
    
    print("\n" + "=" * 60)
    print("🔄 Main.py工作流程分析")
    print("=" * 60)
    
    try:
        # 模拟完整的执行流程
        print("📋 执行流程步骤:")
        
        steps = [
            ("1. 参数解析", "parse_args()", "解析命令行参数"),
            ("2. 随机种子设置", "seed_everything(args.seed)", "确保实验可重现"),
            ("3. 配置加载", "Config.fromfile(args.config)", "加载配置文件"),
            ("4. GPU重置", "reset_gpu()", "清理GPU状态"),
            ("5. 线程设置", "init_num_threads(args.num_threads)", "设置CPU线程数"),
            ("6. 环境信息收集", "collect_env()", "收集系统环境信息"),
            ("7. 注册上下文", "with RegistryContext():", "创建组件注册环境"),
            ("8. 组件构建", "build_from_registry()", "构建训练器、模型、数据模块"),
            ("9. 权重加载", "load_checkpoint()", "加载预训练权重"),
            ("10. 执行训练", "runner.fit()", "开始训练流程")
        ]
        
        for step, function, description in steps:
            print(f"  {step}")
            print(f"    函数: {function}")
            print(f"    说明: {description}")
            print()
        
        # 分析关键设计模式
        print("🎯 关键设计模式:")
        design_patterns = [
            "1. 注册机制模式 - 实现组件的动态构建和加载",
            "2. 配置驱动模式 - 所有组件都通过配置文件定义",
            "3. 状态机模式 - 根据state参数执行不同操作",
            "4. 上下文管理模式 - 使用RegistryContext管理组件生命周期"
        ]
        
        for pattern in design_patterns:
            print(f"  {pattern}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流程分析失败: {e}")
        return False

def show_practice_summary():
    """显示实践总结"""
    
    print("\n" + "=" * 60)
    print("📚 Main.py 实践总结")
    print("=" * 60)
    
    summary_points = [
        "🔧 命令行参数：掌握了参数解析和验证机制",
        "📄 配置加载：理解了配置文件的加载和处理流程",
        "🏗️ 注册机制：学习了组件注册和动态构建",
        "🎯 执行模式：掌握了不同运行模式的配置和执行",
        "💻 命令示例：实践了各种使用场景的命令构建",
        "🔄 工作流程：分析了完整的训练执行过程"
    ]
    
    for point in summary_points:
        print(f"  {point}")
    
    print("\n🎯 关键学习要点:")
    key_points = [
        "1. tools/main.py 是框架的统一入口，简洁而功能强大",
        "2. 配置驱动的设计使得框架高度可定制",
        "3. 注册机制实现了组件的解耦和可扩展性",
        "4. 多种执行模式支持不同的使用场景",
        "5. 完善的错误处理和调试支持"
    ]
    
    for point in key_points:
        print(f"  {point}")
    
    print("\n💡 实践建议:")
    practice_tips = [
        "1. 熟悉各种命令行参数的作用和用法",
        "2. 理解配置文件的层次结构和继承关系",
        "3. 掌握不同执行模式的配置差异",
        "4. 学会使用调试工具和技巧",
        "5. 能够根据需求构建合适的命令行"
    ]
    
    for tip in practice_tips:
        print(f"  {tip}")

def main():
    """主函数"""
    
    print("🎓 LeapAI框架学习 - tools/main.py 详细实践")
    print("本脚本将深入分析框架入口的各个功能模块")
    
    try:
        # 执行分析步骤
        steps = [
            ("分析命令行参数", analyze_command_line_arguments),
            ("分析配置加载", analyze_config_loading),
            ("分析注册机制", analyze_registry_mechanism),
            ("分析执行模式", analyze_execution_modes),
            ("实践命令示例", practice_command_examples),
            ("分析工作流程", analyze_main_workflow)
        ]
        
        completed_steps = 0
        for step_name, step_func in steps:
            print(f"\n🔄 执行步骤: {step_name}")
            if step_func():
                completed_steps += 1
                print(f"✅ {step_name} 完成")
            else:
                print(f"❌ {step_name} 失败")
        
        # 显示实践总结
        show_practice_summary()
        
        print(f"\n🎉 tools/main.py 实践学习完成！")
        print(f"完成步骤: {completed_steps}/{len(steps)}")
        
        return completed_steps == len(steps)
        
    except Exception as e:
        print(f"❌ 实践过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
