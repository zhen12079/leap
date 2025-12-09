#!/usr/bin/env python3
"""
LeapAI框架学习 - 阶段5：多任务训练和拓扑定义实践

本阶段学习目标：
1. 理解多任务训练的设计理念
2. 学习任务拓扑的定义和配置
3. 掌握多任务损失函数和权重平衡
4. 实践多任务训练流程
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def explore_multitask_architecture():
    """探索多任务训练架构"""
    
    print("=" * 60)
    print("🎯 多任务训练架构探索")
    print("=" * 60)
    
    try:
        # 查看多任务相关的配置文件
        config_files = []
        config_dir = "projects/perception/configs"
        
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.py') and 'multi' in file.lower():
                    config_files.append(file)
        
        print(f"📁 发现的多任务配置文件:")
        for i, file in enumerate(config_files, 1):
            print(f"  {i}. {file}")
        
        # 分析主要配置文件中的多任务设置
        main_config = "projects/perception/configs/lpperception_current_hpa_step1.py"
        if os.path.exists(main_config):
            with open(main_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\n📋 分析主配置文件中的多任务设置:")
            
            # 查找多任务相关配置
            multitask_keywords = ['multi_task', 'task', 'topology', 'loss_weight']
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in multitask_keywords):
                    print(f"  第{i+1}行: {line.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多任务架构探索失败: {e}")
        return False

def understand_task_topology():
    """理解任务拓扑定义"""
    
    print("\n" + "=" * 60)
    print("🌐 任务拓扑定义理解")
    print("=" * 60)
    
    try:
        # 查看任务拓扑相关的实现
        topology_files = []
        
        # 搜索可能的拓扑定义文件
        search_paths = [
            "leapai/model/",
            "projects/perception/model/",
            "leapai/data/"
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                if any(keyword in content.lower() for keyword in ['topology', 'task', 'graph']):
                                    rel_path = os.path.relpath(file_path, project_root)
                                    topology_files.append(rel_path)
                            except:
                                continue
        
        print(f"🔍 发现的拓扑相关文件 ({len(topology_files)}个):")
        for i, file in enumerate(topology_files[:10], 1):
            print(f"  {i}. {file}")
        
        if len(topology_files) > 10:
            print(f"     ... 还有 {len(topology_files) - 10} 个文件")
        
        # 分析具体的任务定义
        task_types = ['dynamic', 'static', 'occupancy', 'lane', 'detection']
        print(f"\n📋 感知任务类型:")
        for task_type in task_types:
            task_dir = f"projects/perception/model/head"
            if os.path.exists(task_dir):
                task_files = [f for f in os.listdir(task_dir) if task_type in f.lower()]
                if task_files:
                    print(f"  • {task_type}: {len(task_files)} 个实现")
                    for file in task_files[:2]:
                        print(f"    - {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 任务拓扑理解失败: {e}")
        return False

def analyze_loss_functions():
    """分析多任务损失函数"""
    
    print("\n" + "=" * 60)
    print("📉 多任务损失函数分析")
    print("=" * 60)
    
    try:
        # 查看损失函数实现
        loss_dir = "leapai/model/loss"
        if os.path.exists(loss_dir):
            loss_files = [f for f in os.listdir(loss_dir) if f.endswith('.py')]
            print(f"📁 发现的损失函数文件 ({len(loss_files)}个):")
            for i, file in enumerate(loss_files, 1):
                print(f"  {i}. {file}")
        
        # 分析具体损失函数
        loss_types = ['det', 'seg', 'iou', 'focal', 'cross_entropy']
        print(f"\n📋 损失函数类型分析:")
        
        for loss_type in loss_types:
            found_files = []
            for root, dirs, files in os.walk("leapai/model/loss"):
                for file in files:
                    if file.endswith('.py') and loss_type in file.lower():
                        found_files.append(os.path.join(root, file))
            
            if found_files:
                print(f"  • {loss_type}: {len(found_files)} 个文件")
                for file in found_files[:2]:
                    rel_path = os.path.relpath(file, project_root)
                    print(f"    - {rel_path}")
        
        # 查看损失权重配置
        main_config = "projects/perception/configs/lpperception_current_hpa_step1.py"
        if os.path.exists(main_config):
            with open(main_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\n⚖️ 损失权重配置:")
            lines = content.split('\n')
            for line in lines:
                if 'loss_weight' in line.lower() or 'weight' in line.lower():
                    print(f"  {line.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 损失函数分析失败: {e}")
        return False

def practice_multitask_configuration():
    """实践多任务配置"""
    
    print("\n" + "=" * 60)
    print("⚙️ 多任务配置实践")
    print("=" * 60)
    
    try:
        # 创建一个示例多任务配置
        example_config = {
            "multi_task_config": {
                "dynamic": {
                    "enabled": True,
                    "loss_weight": 1.0,
                    "head_type": "dynamic_head",
                    "output_channels": 256
                },
                "static": {
                    "enabled": True,
                    "loss_weight": 0.5,
                    "head_type": "static_head", 
                    "output_channels": 128
                },
                "occupancy": {
                    "enabled": False,
                    "loss_weight": 0.8,
                    "head_type": "occ_head",
                    "output_channels": 64
                }
            },
            "task_topology": {
                "backbone": "resnet50",
                "neck": "fpn",
                "shared_features": True,
                "task_specific_heads": ["dynamic", "static"]
            }
        }
        
        print("✅ 示例多任务配置创建成功")
        print(f"\n📋 配置内容:")
        
        def print_config(config, indent=0):
            for key, value in config.items():
                if isinstance(value, dict):
                    print("  " * indent + f"• {key}:")
                    print_config(value, indent + 1)
                else:
                    print("  " * indent + f"  {key}: {value}")
        
        print_config(example_config)
        
        # 分析实际配置文件
        actual_config = "projects/perception/configs/lpperception_current_hpa_step1.py"
        if os.path.exists(actual_config):
            with open(actual_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\n🔍 实际配置文件分析:")
            
            # 查找任务配置
            task_configs = []
            lines = content.split('\n')
            in_task_config = False
            
            for line in lines:
                if 'multi_task' in line.lower() or 'task_config' in line.lower():
                    in_task_config = True
                    task_configs.append(line)
                elif in_task_config:
                    if line.strip() == '' or (not line.startswith(' ') and not line.startswith('\t')):
                        in_task_config = False
                    else:
                        task_configs.append(line)
            
            print("任务配置片段:")
            for line in task_configs[:10]:
                print(f"  {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多任务配置实践失败: {e}")
        return False

def simulate_multitask_training():
    """模拟多任务训练流程"""
    
    print("\n" + "=" * 60)
    print("🏃 多任务训练流程模拟")
    print("=" * 60)
    
    try:
        # 创建模拟的多任务模型
        class MultiTaskModel(nn.Module):
            """模拟的多任务模型"""
            
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 共享的backbone
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((32, 32))
                )
                
                # 任务特定的头
                self.dynamic_head = nn.Sequential(
                    nn.Linear(64 * 32 * 32, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)  # 10个动态目标类别
                )
                
                self.static_head = nn.Sequential(
                    nn.Linear(64 * 32 * 32, 128),
                    nn.ReLU(),
                    nn.Linear(128, 5)  # 5个静态地图元素
                )
            
            def forward(self, x):
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
                
                dynamic_output = self.dynamic_head(features)
                static_output = self.static_head(features)
                
                return {
                    'dynamic': dynamic_output,
                    'static': static_output
                }
        
        # 创建模拟数据
        batch_size = 4
        input_data = torch.randn(batch_size, 3, 224, 224)
        dynamic_target = torch.randint(0, 10, (batch_size,))
        static_target = torch.randint(0, 5, (batch_size,))
        
        print("✅ 模拟数据和模型创建成功")
        
        # 创建模型和优化器
        model = MultiTaskModel({})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 定义损失函数
        ce_loss = nn.CrossEntropyLoss()
        
        # 模拟训练步骤
        print("\n🔄 模拟训练步骤:")
        
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_data)
        print(f"  • 动态任务输出形状: {outputs['dynamic'].shape}")
        print(f"  • 静态任务输出形状: {outputs['static'].shape}")
        
        # 计算损失
        dynamic_loss = ce_loss(outputs['dynamic'], dynamic_target)
        static_loss = ce_loss(outputs['static'], static_target)
        
        # 加权总损失
        total_loss = 1.0 * dynamic_loss + 0.5 * static_loss
        
        print(f"  • 动态任务损失: {dynamic_loss.item():.4f}")
        print(f"  • 静态任务损失: {static_loss.item():.4f}")
        print(f"  • 总损失: {total_loss.item():.4f}")
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        print("✅ 多任务训练步骤完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 多任务训练模拟失败: {e}")
        return False

def show_learning_summary():
    """显示学习总结"""
    
    print("\n" + "=" * 60)
    print("📚 阶段5学习总结")
    print("=" * 60)
    
    summary_points = [
        "🎯 多任务架构：理解了多任务训练的设计理念",
        "🌐 任务拓扑：掌握了任务拓扑的定义和配置方法",
        "📉 损失函数：分析了多任务损失函数和权重平衡策略",
        "⚙️ 配置实践：实践了多任务配置的创建和使用",
        "🏃 训练流程：模拟了多任务训练的完整流程"
    ]
    
    for point in summary_points:
        print(f"  {point}")
    
    print("\n🎯 下一步学习建议:")
    next_steps = [
        "1. 学习感知任务的具体实现细节",
        "2. 掌握分布式训练和部署机制",
        "3. 实践完整的训练任务",
        "4. 尝试添加新的感知任务"
    ]
    
    for step in next_steps:
        print(f"  {step}")
    
    print("\n💡 关键概念回顾:")
    key_concepts = [
        "• 多任务学习：单一模型处理多个相关任务",
        "• 任务拓扑：定义任务间的共享和独立部分",
        "• 损失权重：平衡不同任务的重要性",
        "• 特征共享：提高模型效率和泛化能力"
    ]
    
    for concept in key_concepts:
        print(f"  {concept}")

def main():
    """主函数"""
    
    print("🎓 LeapAI框架学习 - 阶段5：多任务训练和拓扑定义")
    print("本阶段将深入理解LeapAI的多任务训练机制")
    
    try:
        # 执行学习步骤
        steps = [
            ("探索多任务训练架构", explore_multitask_architecture),
            ("理解任务拓扑定义", understand_task_topology),
            ("分析多任务损失函数", analyze_loss_functions),
            ("实践多任务配置", practice_multitask_configuration),
            ("模拟多任务训练流程", simulate_multitask_training)
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
        
        print(f"\n🎉 阶段5学习完成！")
        print(f"完成步骤: {completed_steps}/{len(steps)}")
        
        return completed_steps == len(steps)
        
    except Exception as e:
        print(f"❌ 学习过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
