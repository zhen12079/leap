# LeapAI框架快速开始指南

## 🎯 学习目标
通过本指南，您将能够：
- 理解LeapAI框架的核心架构
- 掌握配置系统的使用方法
- 学会运行和修改训练任务
- 具备扩展框架的能力

## 🚀 快速开始

### 第1步：环境准备

```bash
# 1. 设置环境变量
export LEAPAI_TASK_CONFIG=projects/perception/configs/lpperception_current_hpa_step1.py

# 2. 安装依赖（如果需要）
pip install -r requirements.txt

# 3. 验证环境
python -c "from leapai import logger; print('LeapAI导入成功')"
```

### 第2步：运行学习练习

```bash
# 运行第1阶段：框架架构理解
python helloworld/step1_understanding_architecture.py

# 运行第2阶段：配置系统实践
python helloworld/step2_config_system_practice.py

# 运行第3阶段：数据模块实践
python helloworld/step3_data_module_practice.py
```

### 第3步：运行实际训练

```bash
# 调试模式训练（小数据集）
export my_debug=yes
python tools/main.py --config projects/perception/entry.py --state train --with-val

# 完整训练
python tools/main.py --config projects/perception/entry.py --state train --with-val
```

## 📋 学习路径检查清单

### ✅ 阶段1：框架整体理解
- [ ] 理解多任务统一训练的设计理念
- [ ] 掌握节点化模型的思想
- [ ] 了解配置驱动的开发模式
- [ ] 熟悉分布式训练支持

### ✅ 阶段2：配置系统和入口机制
- [ ] 能够修改配置文件参数
- [ ] 理解主配置和子任务配置关系
- [ ] 掌握命令行参数使用
- [ ] 学会调试配置加载问题

### ✅ 阶段3：数据模块和数据处理
- [ ] 理解DataModule的工作原理
- [ ] 掌握多任务数据加载机制
- [ ] 学会数据增强的使用
- [ ] 了解分布式数据采样

### 🔄 阶段4：模型构建和NodeGraph机制
- [ ] 理解节点化模型的设计
- [ ] 掌握拓扑定义方法
- [ ] 学会模型组件的注册
- [ ] 了解多任务协调机制

### 🔄 阶段5：多任务训练和拓扑定义
- [ ] 理解任务拓扑的定义
- [ ] 掌握损失函数的聚合
- [ ] 学会任务权重平衡
- [ ] 了解训练和推理模式切换

### 🔄 阶段6：感知任务的具体实现
- [ ] 理解动态任务（目标检测）
- [ ] 掌握静态任务（地图构建）
- [ ] 了解占用网络实现
- [ ] 学会多传感器融合

### 🔄 阶段7：分布式训练和部署
- [ ] 掌握DDP配置
- [ ] 了解多机训练设置
- [ ] 学会ONNX导出
- [ ] 掌握模型部署流程

## 🛠️ 实践项目

### 项目1：运行完整训练任务
**目标**：成功运行一个感知任务的完整训练

**步骤**：
1. 配置数据路径
2. 修改训练参数
3. 启动训练任务
4. 监控训练过程
5. 分析训练结果

**验证标准**：
- [ ] 训练过程无错误
- [ ] 损失函数正常下降
- [ ] 能够保存检查点
- [ ] 验证过程正常运行

### 项目2：添加新的感知任务
**目标**：在框架中添加一个新的感知任务

**步骤**：
1. 定义任务配置文件
2. 实现数据处理逻辑
3. 设计模型头部
4. 定义拓扑函数
5. 集成到多任务训练

**验证标准**：
- [ ] 新任务能够正常训练
- [ ] 与现有任务无冲突
- [ ] 损失函数计算正确
- [ ] 能够输出预测结果

### 项目3：扩展现有组件
**目标**：修改或扩展现有的模型组件

**步骤**：
1. 选择要修改的组件
2. 分析现有实现
3. 设计新的功能
4. 实现和测试
5. 性能对比分析

**验证标准**：
- [ ] 新功能正常工作
- [ ] 不影响现有功能
- [ ] 性能有所提升
- [ ] 代码质量良好

## 🔧 常用操作指南

### 修改学习率
```python
# 在配置文件中修改
float_lr = 1e-4  # 原值 2e-4
finetune_lr = 5e-5  # 原值 1e-4
```

### 调整Batch Size
```python
# 在配置文件中修改
batch_sizes = {
    "dynamic": {"train": 8, "val": 1},  # 减小batch size
    "static": {"train": 8, "val": 1}
}
```

### 启用/禁用模块
```python
# 在配置文件中修改
enable_lidar = True  # 启用LiDAR
enable_dynamic_temporal = False  # 禁用时序
```

### 调整训练步数
```python
# 在配置文件中修改
max_steps = 5000  # 减少训练步数用于调试
train_steps = 4000
finetune_steps = 1000
```

## 🐛 调试技巧

### 1. 使用调试模式
```bash
export my_debug=yes
# 这将使用小数据集和减少的训练步数
```

### 2. 添加日志输出
```python
from leapai import logger
logger.info("调试信息")
```

### 3. 使用VSCode调试
在`.vscode/launch.json`中配置：
```json
{
    "name": "Debug LeapAI",
    "type": "debugpy",
    "request": "launch",
    "program": "tools/main.py",
    "args": [
        "--config", "projects/perception/entry.py",
        "--state", "train"
    ],
    "env": {
        "LEAPAI_TASK_CONFIG": "projects/perception/configs/lpperception_current_hpa_step1.py"
    }
}
```

### 4. 检查配置加载
```python
# 运行配置调试脚本
python helloworld/debug_config.py
```

## 📖 推荐学习顺序

### 第1周：基础理解
1. 阅读README.md了解框架概览
2. 运行step1_understanding_architecture.py
3. 分析tools/main.py的入口逻辑
4. 理解leapai/registry.py的注册机制

### 第2周：配置和数据
1. 运行step2_config_system_practice.py
2. 修改配置文件并验证
3. 运行step3_data_module_practice.py
4. 理解数据加载流程

### 第3周：模型和训练
1. 分析leapai/model/node_graph.py
2. 理解projects/perception/topology.py
3. 查看动态和静态任务实现
4. 尝试运行完整训练

### 第4周：实践和扩展
1. 完成实践项目1
2. 尝试添加新任务
3. 扩展现有组件
4. 性能优化和调试

## 💡 学习建议

1. **循序渐进**：按照学习路径逐步深入，不要跳跃
2. **实践为主**：每个阶段都要动手操作，理论结合实践
3. **源码阅读**：深入理解实现细节，培养代码阅读能力
4. **问题驱动**：带着具体问题学习，提高学习效率
5. **文档记录**：记录学习心得、问题和解决方案

## 🤝 获取帮助

### 查看文档
- README.md：框架概览
- docs/changelog.md：版本更新记录
- 代码注释：详细的实现说明

### 分析示例
- projects/perception/：完整的感知任务示例
- tests/：测试用例和使用示例
- 配置文件：各种场景的配置示例

### 调试工具
- 日志系统：leapai/logger_utils.py
- 性能分析：leapai/profiler/
- 可视化工具：leapai/visualize/

## 🎯 学习成果检验

完成学习后，您应该能够：

### 基础能力
- [ ] 独立配置和运行训练任务
- [ ] 理解框架的核心设计理念
- [ ] 掌握配置系统的使用方法
- [ ] 能够进行基本的调试

### 进阶能力
- [ ] 添加新的感知任务
- [ ] 修改和扩展现有组件
- [ ] 优化训练性能
- [ ] 进行模型部署

### 专家能力
- [ ] 深度定制框架功能
- [ ] 解决复杂的技术问题
- [ ] 贡献代码到框架
- [ ] 指导其他开发者

---

**祝您学习愉快！如有问题，请参考框架文档或分析现有代码实现。**
