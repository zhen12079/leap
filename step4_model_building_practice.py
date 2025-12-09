#!/usr/bin/env python3
"""
LeapAI框架学习 - 阶段4：模型构建和NodeGraph机制实践

本阶段学习目标：
1. 理解NodeGraph设计理念和架构
2. 学习节点化模型构建方法
3. 掌握模型拓扑定义和连接
4. 实践自定义节点开发
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def explore_node_graph_architecture():
    """探索NodeGraph架构"""
    
    print("=" * 60)
    print("🏗️ NodeGraph架构探索")
    print("=" * 60)
    
    try:
        # 读取NodeGraph核心文件
        node_graph_path = "leapai/model/node_graph.py"
        
        if not os.path.exists(node_graph_path):
            print(f"❌ NodeGraph文件不存在: {node_graph_path}")
            return False
        
        with open(node_graph_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ NodeGraph核心文件读取成功")
        
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
        print(f"❌ NodeGraph架构探索失败: {e}")
        return False

def understand_node_concept():
    """理解节点概念"""
    
    print("\n" + "=" * 60)
    print("🔗 节点概念理解")
    print("=" * 60)
    
    try:
        # 尝试导入NodeGraph相关模块
        from leapai.model.node_graph import NodeGraph, Node
        
        print("✅ NodeGraph模块导入成功")
        
        # 分析Node基类
        if hasattr(Node, '__doc__') and Node.__doc__:
            print(f"\n📖 Node类文档:")
            print(Node.__doc__[:200] + "..." if len(Node.__doc__) > 200 else Node.__doc__)
        
        # 查看Node的方法
        node_methods = [method for method in dir(Node) if not method.startswith('_')]
        print(f"\n🔧 Node类方法:")
        for i, method in enumerate(node_methods, 1):
            print(f"  {i}. {method}")
        
        # 查看NodeGraph的方法
        graph_methods = [method for method in dir(NodeGraph) if not method.startswith('_')]
        print(f"\n🔧 NodeGraph类方法:")
        for i, method in enumerate(graph_methods[:10], 1):  # 只显示前10个
            print(f"  {i}. {method}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("💡 可能需要先完成环境配置")
        return False
    except Exception as e:
        print(f"❌ 节点概念理解失败: {e}")
        return False

def analyze_model_topology():
    """分析模型拓扑结构"""
    
    print("\n" + "=" * 60)
    print("🌐 模型拓扑结构分析")
    print("=" * 60)
    
    try:
        # 查看perception项目的模型配置
        model_configs = []
        perception_model_dir = "projects/perception/model"
        
        if os.path.exists(perception_model_dir):
            for root, dirs, files in os.walk(perception_model_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        rel_path = os.path.relpath(os.path.join(root, file), perception_model_dir)
                        model_configs.append(rel_path)
        
        print(f"📁 发现的模型文件 ({len(model_configs)}个):")
        for i, config in enumerate(model_configs[:15], 1):  # 只显示前15个
            print(f"  {i:2d}. {config}")
        
        if len(model_configs) > 15:
            print(f"     ... 还有 {len(model_configs) - 15} 个文件")
        
        # 分析关键模型组件
        key_components = [
            "backbone", "neck", "head", "fusion", "task_module"
        ]
        
        print(f"\n🏗️ 关键模型组件:")
        for component in key_components:
            component_dir = f"projects/perception/model/{component}"
            if os.path.exists(component_dir):
                files = [f for f in os.listdir(component_dir) if f.endswith('.py')]
                print(f"  • {component}: {len(files)} 个文件")
                for file in files[:3]:  # 只显示前3个
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    - ... 还有 {len(files) - 3} 个文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型拓扑分析失败: {e}")
        return False

def practice_node_creation():
    """实践节点创建"""
    
    print("\n" + "=" * 60)
    print("🛠️ 节点创建实践")
    print("=" * 60)
    
    try:
        # 尝试导入必要的模块
        from leapai.model.node_graph import Node, NodeGraph
        from leapai.registry import RegistryContext, build_from_registry
        
        print("✅ 模块导入成功")
        
        # 创建一个简单的自定义节点示例
        class SimpleConvNode(Node):
            """简单的卷积节点示例"""
            
            def __init__(self, in_channels, out_channels, kernel_size=3):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
                self.bn = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                return self.relu(self.bn(self.conv(x)))
        
        class SimplePoolingNode(Node):
            """简单的池化节点示例"""
            
            def __init__(self, pool_type='max', kernel_size=2):
                super().__init__()
                if pool_type == 'max':
                    self.pool = nn.MaxPool2d(kernel_size)
                else:
                    self.pool = nn.AvgPool2d(kernel_size)
            
            def forward(self, x):
                return self.pool(x)
        
        print("✅ 自定义节点类创建成功")
        
        # 测试节点功能
        with torch.no_grad():
            # 创建测试数据
            test_input = torch.randn(1, 64, 32, 32)
            
            # 创建并测试卷积节点
            conv_node = SimpleConvNode(64, 128)
            conv_output = conv_node(test_input)
            print(f"✅ 卷积节点测试成功: {test_input.shape} -> {conv_output.shape}")
            
            # 创建并测试池化节点
            pool_node = SimplePoolingNode('max', 2)
            pool_output = pool_node(conv_output)
            print(f"✅ 池化节点测试成功: {conv_output.shape} -> {pool_output.shape}")
        
        print("✅ 节点创建实践完成")
        return True
        
    except Exception as e:
        print(f"❌ 节点创建实践失败: {e}")
        return False

def analyze_existing_models():
    """分析现有模型实现"""
    
    print("\n" + "=" * 60)
    print("🔍 现有模型实现分析")
    print("=" * 60)
    
    try:
        # 分析perception项目的模型基类
        model_base_path = "projects/perception/model_base.py"
        
        if os.path.exists(model_base_path):
            with open(model_base_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("✅ 模型基类文件读取成功")
            
            # 提取关键类和方法
            lines = content.split('\n')
            classes = []
            methods = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('class '):
                    class_name = line.split('(')[0].replace('class ', '').strip(':')
                    classes.append(class_name)
                elif line.startswith('    def ') and not line.startswith('    def _'):
                    method_name = line.split('(')[0].strip().replace('def ', '')
                    methods.append(method_name)
            
            print(f"\n📋 发现的模型类:")
            for i, cls in enumerate(classes, 1):
                print(f"  {i}. {cls}")
            
            print(f"\n📋 发现的公共方法:")
            for i, method in enumerate(methods[:8], 1):
                print(f"  {i}. {method}")
        
        # 分析具体的模型头实现
        head_dir = "projects/perception/model/head"
        if os.path.exists(head_dir):
            head_files = [f for f in os.listdir(head_dir) if f.endswith('.py')]
            print(f"\n🎯 模型头实现 ({len(head_files)}个):")
            for i, file in enumerate(head_files, 1):
                print(f"  {i}. {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 现有模型分析失败: {e}")
        return False

def practice_model_configuration():
    """实践模型配置"""
    
    print("\n" + "=" * 60)
    print("⚙️ 模型配置实践")
    print("=" * 60)
    
    try:
        # 分析配置文件中的模型定义
        config_path = "projects/perception/configs/lpperception_current_hpa_step1.py"
        
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ 配置文件读取成功")
        
        # 查找模型相关配置
        model_sections = []
        lines = content.split('\n')
        current_section = []
        in_model_section = False
        
        for line in lines:
            if any(keyword in line for keyword in ['graph_model', 'model', 'backbone', 'neck', 'head']):
                in_model_section = True
                current_section = [line]
            elif in_model_section:
                if line.strip() == '' or (line.startswith(' ') == False and not line.startswith('\t')):
                    if current_section:
                        model_sections.append('\n'.join(current_section))
                        current_section = []
                    in_model_section = False
                else:
                    current_section.append(line)
        
        if current_section:
            model_sections.append('\n'.join(current_section))
        
        print(f"\n📋 发现的模型配置段 ({len(model_sections)}个):")
        for i, section in enumerate(model_sections, 1):
            lines = section.split('\n')
            title = lines[0].strip() if lines else "Unknown"
            print(f"  {i}. {title}")
            # 显示前几行内容
            for line in lines[1:4]:
                if line.strip():
                    print(f"     {line.strip()}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ 模型配置实践失败: {e}")
        return False

def show_learning_summary():
    """显示学习总结"""
    
    print("\n" + "=" * 60)
    print("📚 阶段4学习总结")
    print("=" * 60)
    
    summary_points = [
        "🏗️ NodeGraph架构：理解了节点化模型的设计理念",
        "🔗 节点概念：掌握了Node基类和节点连接机制",
        "🌐 模型拓扑：分析了模型的层次结构和组件关系",
        "🛠️ 节点创建：实践了自定义节点的开发",
        "🔍 模型分析：深入了解了现有模型的实现方式",
        "⚙️ 配置系统：掌握了模型配置的定义和使用"
    ]
    
    for point in summary_points:
        print(f"  {point}")
    
    print("\n🎯 下一步学习建议:")
    next_steps = [
        "1. 深入理解多任务训练机制",
        "2. 学习感知任务的具体实现",
        "3. 掌握分布式训练和部署",
        "4. 实践完整的训练流程"
    ]
    
    for step in next_steps:
        print(f"  {step}")
    
    print("\n💡 关键文件回顾:")
    key_files = [
        "• leapai/model/node_graph.py - NodeGraph核心实现",
        "• projects/perception/model_base.py - 模型基类",
        "• projects/perception/model/ - 具体模型实现",
        "• projects/perception/configs/ - 模型配置文件"
    ]
    
    for file in key_files:
        print(f"  {file}")

def main():
    """主函数"""
    
    print("🎓 LeapAI框架学习 - 阶段4：模型构建和NodeGraph机制")
    print("本阶段将深入理解LeapAI的节点化模型架构")
    
    try:
        # 执行学习步骤
        steps = [
            ("探索NodeGraph架构", explore_node_graph_architecture),
            ("理解节点概念", understand_node_concept),
            ("分析模型拓扑结构", analyze_model_topology),
            ("实践节点创建", practice_node_creation),
            ("分析现有模型实现", analyze_existing_models),
            ("实践模型配置", practice_model_configuration)
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
        
        print(f"\n🎉 阶段4学习完成！")
        print(f"完成步骤: {completed_steps}/{len(steps)}")
        
        return completed_steps == len(steps)
        
    except Exception as e:
        print(f"❌ 学习过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
