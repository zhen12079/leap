#!/usr/bin/env python3
"""
LeapAI框架学习 - 第1步：理解框架整体架构

这个练习将帮助您：
1. 理解框架的核心组件
2. 学习配置系统的使用
3. 掌握入口机制的工作原理
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def step1_understand_framework_architecture():
    """第1步：理解框架整体架构"""
    
    print("=" * 60)
    print("🏗️  LeapAI框架架构理解")
    print("=" * 60)
    
    # 1. 理解项目结构
    print("\n📁 1. 项目核心结构分析：")
    print("├── leapai/           # 框架核心库")
    print("│   ├── data/        # 数据处理模块")
    print("│   ├── model/       # 模型组件")
    print("│   ├── callback/    # 回调函数")
    print("│   └── utils/       # 工具函数")
    print("├── projects/        # 项目配置")
    print("│   ├── perception/  # 感知任务")
    print("│   ├── APA/         # APA任务")
    print("│   └── FUSION/      # 融合任务")
    print("├── tools/           # 工具脚本")
    print("└── tests/           # 测试用例")
    
    # 2. 理解核心设计理念
    print("\n💡 2. 核心设计理念：")
    design_concepts = {
        "多任务统一训练": "在一个框架中处理动态检测、静态地图、占用网络等多个任务",
        "节点化模型": "使用NodeGraph将模型分解为可复用的节点",
        "配置驱动": "通过Python配置文件控制实验参数和模型结构",
        "分布式原生": "内置多机多卡DDP训练支持",
        "模块化设计": "每个组件都可以独立替换和扩展"
    }
    
    for concept, description in design_concepts.items():
        print(f"  • {concept}: {description}")
    
    # 3. 分析关键文件
    print("\n📄 3. 关键文件分析：")
    key_files = {
        "tools/main.py": "统一入口，处理命令行参数和训练流程",
        "leapai/registry.py": "组件注册机制，支持动态构建",
        "leapai/utils/config.py": "配置系统，支持Python和YAML配置",
        "leapai/model/node_graph.py": "节点化模型实现",
        "leapai/data/data_module.py": "数据模块，支持多任务数据加载"
    }
    
    for file_path, description in key_files.items():
        print(f"  • {file_path}: {description}")
    
    return True

def step2_explore_entry_mechanism():
    """第2步：探索入口机制"""
    
    print("\n" + "=" * 60)
    print("🚀 入口机制探索")
    print("=" * 60)
    
    # 分析main.py的核心逻辑
    print("\n📝 分析 tools/main.py 的核心逻辑：")
    
    main_logic = [
        "1. 参数解析 (parse_args)",
        "2. 配置加载 (Config.fromfile)",
        "3. 环境初始化 (seed_everything, reset_gpu)",
        "4. 组件构建 (build_from_registry)",
        "5. 训练器初始化 (Lightning Trainer)",
        "6. 训练/验证/测试执行"
    ]
    
    for step in main_logic:
        print(f"  {step}")
    
    # 模拟配置加载
    print("\n⚙️  配置系统演示：")
    
    # 创建一个简单的配置示例
    sample_config = """
# 示例配置文件
job_name = "demo_experiment"
max_steps = 1000
float_lr = 2e-4

# 多任务配置
multi_task_config = {
    "dynamic": "projects/perception/dynamic.py",
    "static": "projects/perception/static.py"
}

# 数据配置
batch_sizes = {
    "dynamic": {"train": 16, "val": 1},
    "static": {"train": 16, "val": 1}
}
"""
    
    print("配置文件示例：")
    print(sample_config)
    
    return True

def step3_understand_registry_system():
    """第3步：理解注册系统"""
    
    print("\n" + "=" * 60)
    print("📋 注册系统理解")
    print("=" * 60)
    
    print("\n🔧 注册系统的作用：")
    registry_benefits = [
        "解耦组件定义和使用",
        "支持动态组件加载",
        "便于单元测试",
        "支持配置文件驱动"
    ]
    
    for benefit in registry_benefits:
        print(f"  • {benefit}")
    
    print("\n📝 注册系统使用示例：")
    registry_example = '''
# 1. 定义组件
@LEAP_OBJECTS.register_module()
class MyDataModule(L.LightningDataModule):
    def __init__(self, ...):
        pass

# 2. 在配置中使用
data_module = dict(
    type=MyDataModule,  # 或 "MyDataModule"
    batch_size=16,
    num_workers=4
)

# 3. 构建组件
data_module = build_from_registry(data_module)
'''
    
    print(registry_example)
    
    return True

def step4_analyze_multitask_structure():
    """第4步：分析多任务结构"""
    
    print("\n" + "=" * 60)
    print("🎯 多任务结构分析")
    print("=" * 60)
    
    print("\n🔄 多任务训练流程：")
    multitask_flow = [
        "1. 各任务独立数据加载",
        "2. CombinedLoader合并数据流",
        "3. 共享骨干网络提取特征",
        "4. 任务特定头部处理",
        "5. 损失计算和权重平衡",
        "6. 反向传播和参数更新"
    ]
    
    for step in multitask_flow:
        print(f"  {step}")
    
    print("\n📊 任务类型分析：")
    task_types = {
        "动态任务": "3D目标检测 (车辆、行人、障碍物等)",
        "静态任务": "地图构建 (车道线、路缘、标识等)",
        "占用网络": "3D场景占用预测",
        "融合任务": "多传感器数据融合"
    }
    
    for task_type, description in task_types.items():
        print(f"  • {task_type}: {description}")
    
    return True

def interactive_quiz():
    """互动问答环节"""
    
    print("\n" + "=" * 60)
    print("🤔 互动问答")
    print("=" * 60)
    
    questions = [
        {
            "question": "LeapAI框架的核心设计理念是什么？",
            "answer": "多任务统一训练、节点化模型、配置驱动、分布式原生、模块化设计"
        },
        {
            "question": "框架的统一入口文件是什么？",
            "answer": "tools/main.py"
        },
        {
            "question": "注册系统的主要作用是什么？",
            "answer": "解耦组件定义和使用，支持动态组件加载"
        },
        {
            "question": "多任务训练中如何处理不同任务的数据？",
            "answer": "使用CombinedLoader合并多个任务的数据流"
        }
    ]
    
    print("\n💡 思考题（请先自己思考，再看答案）：")
    
    for i, q in enumerate(questions, 1):
        print(f"\n问题{i}: {q['question']}")
        input("按回车键查看答案...")
        print(f"答案: {q['answer']}")
    
    return True

def main():
    """主函数"""
    
    print("🎓 欢迎来到LeapAI框架学习课程！")
    print("本练习将帮助您理解框架的整体架构和设计理念")
    
    try:
        # 执行学习步骤
        step1_understand_framework_architecture()
        step2_explore_entry_mechanism()
        step3_understand_registry_system()
        step4_analyze_multitask_structure()
        interactive_quiz()
        
        print("\n" + "=" * 60)
        print("🎉 第1阶段学习完成！")
        print("=" * 60)
        
        print("\n📋 下一步学习建议：")
        next_steps = [
            "1. 深入阅读 tools/main.py 源码",
            "2. 分析 leapai/registry.py 的实现",
            "3. 查看 projects/perception/configs/ 中的配置示例",
            "4. 尝试修改配置文件并运行训练"
        ]
        
        for step in next_steps:
            print(f"  {step}")
        
        print("\n💡 记住：理论学习要结合实践，多动手操作才能深入理解！")
        
    except Exception as e:
        print(f"❌ 学习过程中出现错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
