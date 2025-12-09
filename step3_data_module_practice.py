#!/usr/bin/env python3
"""
LeapAI框架学习 - 阶段3：数据模块和数据处理流程实践

本阶段学习目标：
1. 理解多任务数据加载机制
2. 学习数据预处理和增强流程
3. 掌握目标生成和标签处理
4. 实践数据模块的配置和使用
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def explore_data_module_architecture():
    """探索数据模块架构"""
    
    print("=" * 60)
    print("📊 数据模块架构探索")
    print("=" * 60)
    
    try:
        # 读取数据模块核心文件
        data_module_path = "leapai/data/data_module.py"
        
        if not os.path.exists(data_module_path):
            print(f"❌ 数据模块文件不存在: {data_module_path}")
            return False
        
        with open(data_module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ 数据模块核心文件读取成功")
        
        # 分析关键类和方法
        key_classes = []
        key_methods = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('class '):
                class_name = line.split('(')[0].replace('class ', '').strip(':')
                key_classes.append(class_name)
            elif 'def ' in line and not line.startswith('#'):
                method_name = line.split('(')[0].strip().replace('def ', '')
                if not method_name.startswith('_'):
                    key_methods.append(method_name)
        
        print(f"\n📋 发现的关键类:")
        for i, cls in enumerate(key_classes, 1):
            print(f"  {i}. {cls}")
        
        print(f"\n📋 发现的公共方法:")
        for i, method in enumerate(key_methods[:10], 1):  # 只显示前10个
            print(f"  {i}. {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据模块架构探索失败: {e}")
        return False

def analyze_dataloader_components():
    """分析数据加载器组件"""
    
    print("\n" + "=" * 60)
    print("🔄 数据加载器组件分析")
    print("=" * 60)
    
    try:
        # 查看数据加载器目录
        dataloader_dir = "leapai/data/dataloader"
        
        if os.path.exists(dataloader_dir):
            dataloader_files = [f for f in os.listdir(dataloader_dir) if f.endswith('.py')]
            print(f"📁 数据加载器文件 ({len(dataloader_files)}个):")
            for i, file in enumerate(dataloader_files, 1):
                print(f"  {i}. {file}")
        
        # 分析具体的数据加载器实现
        key_loaders = [
            "combined_dataloader.py",
            "cycle_iterator.py"
        ]
        
        print(f"\n🔍 关键数据加载器分析:")
        for loader in key_loaders:
            loader_path = f"leapai/data/dataloader/{loader}"
            if os.path.exists(loader_path):
                with open(loader_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取类信息
                lines = content.split('\n')
                classes = []
                for line in lines:
                    if line.strip().startswith('class '):
                        class_name = line.split('(')[0].replace('class ', '').strip(':')
                        classes.append(class_name)
                
                print(f"  • {loader}: {classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器分析失败: {e}")
        return False

def explore_dataset_implementations():
    """探索数据集实现"""
    
    print("\n" + "=" * 60)
    print("📂 数据集实现探索")
    print("=" * 60)
    
    try:
        # 查看数据集目录
        dataset_dir = "leapai/data/dataset"
        
        if os.path.exists(dataset_dir):
            dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.py')]
            print(f"📁 数据集文件 ({len(dataset_files)}个):")
            for i, file in enumerate(dataset_files, 1):
                print(f"  {i}. {file}")
        
        # 分析关键数据集类型
        dataset_types = {
            "bev_dataset.py": "BEV数据集",
            "fusion_dataset.py": "融合数据集", 
            "lidar_dataset.py": "LiDAR数据集",
            "video_iterable_dataset.py": "视频数据集"
        }
        
        print(f"\n🎯 关键数据集类型:")
        for filename, description in dataset_types.items():
            filepath = f"leapai/data/dataset/{filename}"
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取类信息
                lines = content.split('\n')
                classes = []
                for line in lines:
                    if line.strip().startswith('class '):
                        class_name = line.split('(')[0].replace('class ', '').strip(':')
                        classes.append(class_name)
                
                print(f"  • {description} ({filename}): {classes}")
        
        # 查看perception项目的数据集实现
        perception_dataset_dir = "projects/perception/dataset"
        if os.path.exists(perception_dataset_dir):
            perception_files = [f for f in os.listdir(perception_dataset_dir) if f.endswith('.py')]
            print(f"\n🚗 Perception项目数据集 ({len(perception_files)}个):")
            for i, file in enumerate(perception_files, 1):
                print(f"  {i}. {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集探索失败: {e}")
        return False

def analyze_data_transforms():
    """分析数据变换"""
    
    print("\n" + "=" * 60)
    print("🔄 数据变换分析")
    print("=" * 60)
    
    try:
        # 查看数据变换目录
        transform_dir = "leapai/data/transform"
        
        if os.path.exists(transform_dir):
            transform_files = [f for f in os.listdir(transform_dir) if f.endswith('.py')]
            print(f"📁 数据变换文件 ({len(transform_files)}个):")
            for i, file in enumerate(transform_files, 1):
                print(f"  {i}. {file}")
        
        # 分析关键变换类型
        transform_types = {
            "augment.py": "数据增强",
            "lidar_augment.py": "LiDAR增强",
            "image_tensor_transfer.py": "图像变换",
            "lidar_processor.py": "LiDAR处理",
            "point2voxel.py": "点云体素化"
        }
        
        print(f"\n🔧 关键变换类型:")
        for filename, description in transform_types.items():
            filepath = f"leapai/data/transform/{filename}"
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 统计函数数量
                lines = content.split('\n')
                functions = []
                for line in lines:
                    if line.strip().startswith('def '):
                        func_name = line.split('(')[0].strip().replace('def ', '')
                        if not func_name.startswith('_'):
                            functions.append(func_name)
                
                print(f"  • {description} ({filename}): {len(functions)} 个函数")
                for func in functions[:3]:  # 只显示前3个
                    print(f"    - {func}")
                if len(functions) > 3:
                    print(f"    - ... 还有 {len(functions) - 3} 个函数")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据变换分析失败: {e}")
        return False

def explore_target_generation():
    """探索目标生成"""
    
    print("\n" + "=" * 60)
    print("🎯 目标生成探索")
    print("=" * 60)
    
    try:
        # 查看目标生成目录
        target_dir = "leapai/data/target"
        
        if os.path.exists(target_dir):
            target_files = [f for f in os.listdir(target_dir) if f.endswith('.py')]
            print(f"📁 目标生成文件 ({len(target_files)}个):")
            for i, file in enumerate(target_files, 1):
                print(f"  {i}. {file}")
        
        # 分析目标类型
        target_types = {
            "bev_dynamic_target.py": "BEV动态目标",
            "bev_static_target.py": "BEV静态目标",
            "lidar_det_target.py": "LiDAR检测目标",
            "lidar_lane_target.py": "LiDAR车道线目标",
            "lidar_seg_target.py": "LiDAR分割目标"
        }
        
        print(f"\n🎯 目标类型分析:")
        for filename, description in target_types.items():
            filepath = f"leapai/data/target/{filename}"
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取类信息
                lines = content.split('\n')
                classes = []
                for line in lines:
                    if line.strip().startswith('class '):
                        class_name = line.split('(')[0].replace('class ', '').strip(':')
                        classes.append(class_name)
                
                print(f"  • {description} ({filename}): {classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ 目标生成探索失败: {e}")
        return False

def practice_data_configuration():
    """实践数据配置"""
    
    print("\n" + "=" * 60)
    print("⚙️ 数据配置实践")
    print("=" * 60)
    
    try:
        # 分析配置文件中的数据配置
        config_path = "projects/perception/configs/lpperception_current_hpa_step1.py"
        
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ 配置文件读取成功")
        
        # 查找数据相关配置
        data_sections = []
        lines = content.split('\n')
        current_section = []
        in_data_section = False
        
        for line in lines:
            if any(keyword in line for keyword in ['data_module', 'dataset', 'dataloader', 'transform']):
                in_data_section = True
                current_section = [line]
            elif in_data_section:
                if line.strip() == '' or (line.startswith(' ') == False and not line.startswith('\t')):
                    if current_section:
                        data_sections.append('\n'.join(current_section))
                        current_section = []
                    in_data_section = False
                else:
                    current_section.append(line)
        
        if current_section:
            data_sections.append('\n'.join(current_section))
        
        print(f"\n📋 发现的数据配置段 ({len(data_sections)}个):")
        for i, section in enumerate(data_sections, 1):
            lines = section.split('\n')
            title = lines[0].strip() if lines else "Unknown"
            print(f"  {i}. {title}")
            # 显示前几行内容
            for line in lines[1:4]:
                if line.strip():
                    print(f"     {line.strip()}")
            print()
        
        # 创建示例数据配置
        example_config = {
            "data_module": {
                "type": "MultiTaskDataModule",
                "dataset_cfg": {
                    "type": "FusionDataset",
                    "data_root": "/path/to/data",
                    "train_split": "train.txt",
                    "val_split": "val.txt"
                },
                "dataloader_cfg": {
                    "batch_size": 8,
                    "num_workers": 4,
                    "pin_memory": True
                },
                "transform_cfg": {
                    "train": ["RandomFlip", "RandomScale", "Normalize"],
                    "val": ["Normalize"]
                }
            }
        }
        
        print("✅ 示例数据配置:")
        def print_config(config, indent=0):
            for key, value in config.items():
                if isinstance(value, dict):
                    print("  " * indent + f"• {key}:")
                    print_config(value, indent + 1)
                else:
                    print("  " * indent + f"  {key}: {value}")
        
        print_config(example_config)
        
        return True
        
    except Exception as e:
        print(f"❌ 数据配置实践失败: {e}")
        return False

def simulate_data_pipeline():
    """模拟数据流水线"""
    
    print("\n" + "=" * 60)
    print("🔄 数据流水线模拟")
    print("=" * 60)
    
    try:
        # 创建模拟的数据处理流程
        class MockDataPipeline:
            """模拟的数据处理流水线"""
            
            def __init__(self):
                self.steps = [
                    "数据加载",
                    "数据预处理", 
                    "数据增强",
                    "目标生成",
                    "批次组织"
                ]
            
            def process_batch(self, batch_size=4):
                """模拟处理一个批次的数据"""
                print(f"🔄 处理批次大小: {batch_size}")
                
                for i, step in enumerate(self.steps, 1):
                    print(f"  {i}. {step}...")
                    # 模拟处理时间
                    import time
                    time.sleep(0.1)
                    print(f"     ✅ {step}完成")
                
                # 模拟输出数据
                mock_output = {
                    "images": torch.randn(batch_size, 3, 224, 224),
                    "lidar": torch.randn(batch_size, 1000, 4),
                    "targets": {
                        "dynamic": torch.randint(0, 10, (batch_size, 50)),
                        "static": torch.randint(0, 5, (batch_size, 100))
                    },
                    "metadata": ["frame_001", "frame_002", "frame_003", "frame_004"]
                }
                
                return mock_output
        
        # 创建并运行数据流水线
        pipeline = MockDataPipeline()
        output = pipeline.process_batch()
        
        print(f"\n📊 输出数据结构:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  • {key}: {value.shape} {value.dtype}")
            elif isinstance(value, dict):
                print(f"  • {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    - {sub_key}: {sub_value.shape} {sub_value.dtype}")
            else:
                print(f"  • {key}: {type(value).__name__} ({len(value)} 项)")
        
        print("✅ 数据流水线模拟完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据流水线模拟失败: {e}")
        return False

def show_learning_summary():
    """显示学习总结"""
    
    print("\n" + "=" * 60)
    print("📚 阶段3学习总结")
    print("=" * 60)
    
    summary_points = [
        "📊 数据模块架构：理解了多任务数据加载的设计理念",
        "🔄 数据加载器：掌握了多种数据加载器的实现和使用",
        "📂 数据集实现：学习了不同类型数据集的处理方式",
        "🔄 数据变换：掌握了数据预处理和增强技术",
        "🎯 目标生成：理解了标签生成和目标处理机制",
        "⚙️ 配置实践：实践了数据模块的配置和使用"
    ]
    
    for point in summary_points:
        print(f"  {point}")
    
    print("\n🎯 下一步学习建议:")
    next_steps = [
        "1. 学习模型构建和NodeGraph机制",
        "2. 理解多任务训练和拓扑定义",
        "3. 掌握感知任务的具体实现",
        "4. 学习分布式训练和部署"
    ]
    
    for step in next_steps:
        print(f"  {step}")
    
    print("\n💡 关键文件回顾:")
    key_files = [
        "• leapai/data/data_module.py - 数据模块核心",
        "• leapai/data/dataloader/ - 数据加载器实现",
        "• leapai/data/dataset/ - 数据集实现",
        "• leapai/data/transform/ - 数据变换实现",
        "• leapai/data/target/ - 目标生成实现"
    ]
    
    for file in key_files:
        print(f"  {file}")

def main():
    """主函数"""
    
    print("🎓 LeapAI框架学习 - 阶段3：数据模块和数据处理流程")
    print("本阶段将深入理解LeapAI的数据处理机制")
    
    try:
        # 执行学习步骤
        steps = [
            ("探索数据模块架构", explore_data_module_architecture),
            ("分析数据加载器组件", analyze_dataloader_components),
            ("探索数据集实现", explore_dataset_implementations),
            ("分析数据变换", analyze_data_transforms),
            ("探索目标生成", explore_target_generation),
            ("实践数据配置", practice_data_configuration),
            ("模拟数据流水线", simulate_data_pipeline)
        ]
        
        completed_steps = 0
        for step_name, step_func in steps:
            print(f"\n🔄 执行步骤: {step_name}")
            if step_func():
                completed_steps += 1
                print(f"✅ {step_name} 完成")
            else:
                print(f"❌ {step_name} 失败")
        
        # 显示学习总结
        show_learning_summary()
        
        print(f"\n🎉 阶段3学习完成！")
        print(f"完成步骤: {completed_steps}/{len(steps)}")
        
        return completed_steps == len(steps)
        
    except Exception as e:
        print(f"❌ 学习过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
