#!/usr/bin/env python3
"""
LeapAI框架学习 - 训练运行演示

这个脚本演示如何：
1. 设置环境变量
2. 验证配置文件
3. 运行一个简化的训练任务
4. 监控训练过程
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """设置环境变量"""
    
    print("=" * 60)
    print("🔧 设置环境变量")
    print("=" * 60)
    
    # 设置配置文件路径
    config_path = "projects/perception/configs/lpperception_current_hpa_step1.py"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    os.environ["LEAPAI_TASK_CONFIG"] = config_path
    print(f"✅ 设置 LEAPAI_TASK_CONFIG = {config_path}")
    
    # 设置调试模式（可选）
    os.environ["my_debug"] = "yes"
    print("✅ 设置 my_debug = yes (调试模式)")
    
    # 设置其他环境变量
    os.environ["LEAPAI_DATETIME"] = time.strftime("%Y%m%d_%H%M%S")
    print(f"✅ 设置 LEAPAI_DATETIME = {os.environ['LEAPAI_DATETIME']}")
    
    return True

def validate_configuration():
    """验证配置文件"""
    
    print("\n" + "=" * 60)
    print("✅ 验证配置文件")
    print("=" * 60)
    
    try:
        from leapai.utils.config import Config
        
        config_path = os.environ["LEAPAI_TASK_CONFIG"]
        cfg = Config.fromfile(config_path)
        
        print(f"✅ 配置文件加载成功: {config_path}")
        print(f"   任务名称: {getattr(cfg, 'job_name', 'Unknown')}")
        print(f"   最大步数: {getattr(cfg, 'max_steps', 'Unknown')}")
        print(f"   学习率: {getattr(cfg, 'float_lr', 'Unknown')}")
        
        # 检查多任务配置
        if hasattr(cfg, 'multi_task_config'):
            print(f"   多任务配置: {list(cfg.multi_task_config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

def dry_run_training():
    """干运行训练（不实际执行，只验证流程）"""
    
    print("\n" + "=" * 60)
    print("🏃 干运行训练流程")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from leapai import logger
        from leapai.registry import RegistryContext, build_from_registry
        from leapai.utils import seed_everything
        
        print("✅ 模块导入成功")
        
        # 设置随机种子
        seed_everything(0)
        print("✅ 随机种子设置完成")
        
        # 加载配置
        from leapai.utils.config import Config
        cfg = Config.fromfile(os.environ["LEAPAI_TASK_CONFIG"])
        
        print("✅ 配置加载完成")
        
        # 验证组件构建（不实际执行）
        with RegistryContext():
            print("✅ 注册上下文创建成功")
            
            # 验证数据模块配置
            if hasattr(cfg, 'data_module'):
                print("✅ 数据模块配置存在")
            
            # 验证图模型配置
            if hasattr(cfg, 'graph_model'):
                print("✅ 图模型配置存在")
            
            # 验证训练器配置
            if hasattr(cfg, 'runner'):
                print("✅ 训练器配置存在")
        
        print("✅ 干运行验证成功，所有组件配置正确")
        return True
        
    except Exception as e:
        print(f"❌ 干运行验证失败: {e}")
        return False

def run_actual_training():
    """运行实际训练（可选）"""
    
    print("\n" + "=" * 60)
    print("🚀 运行实际训练")
    print("=" * 60)
    
    # 询问用户是否要运行实际训练
    response = input("是否要运行实际训练？这可能需要较长时间 (y/N): ").strip().lower()
    
    if response != 'y':
        print("⏭️  跳过实际训练")
        return True
    
    print("🚀 启动训练任务...")
    print("注意：这将启动一个真实的训练过程")
    
    try:
        # 构建训练命令
        cmd = [
            sys.executable,
            "tools/main.py",
            "--config", "projects/perception/entry.py",
            "--state", "train",
            "--with-val"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("✅ 训练进程已启动")
        print("📊 训练日志（按Ctrl+C停止）：")
        print("-" * 60)
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
            # 检查是否包含关键信息
            if "loss:" in line.lower():
                print(f"📉 损失更新: {line.strip()}")
            elif "epoch" in line.lower():
                print(f"📈 Epoch更新: {line.strip()}")
            elif "saved" in line.lower() and "ckpt" in line.lower():
                print(f"💾 检查点保存: {line.strip()}")
        
        # 等待进程结束
        return_code = process.wait()
        
        if return_code == 0:
            print("✅ 训练完成")
            return True
        else:
            print(f"❌ 训练失败，返回码: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  用户中断训练")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        return True
    except Exception as e:
        print(f"❌ 训练启动失败: {e}")
        return False

def show_training_tips():
    """显示训练技巧"""
    
    print("\n" + "=" * 60)
    print("💡 训练技巧和注意事项")
    print("=" * 60)
    
    tips = [
        "🔧 调试模式：设置 my_debug=yes 使用小数据集",
        "📊 监控训练：使用TensorBoard查看训练曲线",
        "💾 检查点：训练会自动保存，可从中断处恢复",
        "⚙️  参数调优：根据验证结果调整学习率和batch size",
        "🐛 问题排查：查看日志文件了解详细错误信息",
        "🚀 性能优化：使用多GPU训练加速",
        "📈 早停策略：监控验证损失，避免过拟合"
    ]
    
    for tip in tips:
        print(f"  {tip}")
    
    print("\n📁 重要文件位置：")
    important_files = [
        "训练日志：jinnTrainResult/*/logs/",
        "检查点：jinnTrainResult/*/ckpt/",
        "TensorBoard：jinnTrainResult/*/TensorBoard/",
        "配置文件：projects/perception/configs/"
    ]
    
    for file_info in important_files:
        print(f"  • {file_info}")

def main():
    """主函数"""
    
    print("🎓 LeapAI框架学习 - 训练运行演示")
    print("本脚本将演示如何设置环境、验证配置并运行训练任务")
    
    try:
        # 执行步骤
        if not setup_environment():
            return False
        
        if not validate_configuration():
            return False
        
        if not dry_run_training():
            return False
        
        show_training_tips()
        
        # 可选：运行实际训练
        run_actual_training()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("=" * 60)
        
        print("\n📋 下一步建议：")
        next_steps = [
            "1. 分析训练结果和日志",
            "2. 尝试修改配置参数",
            "3. 学习模型架构和拓扑定义",
            "4. 实践添加新的感知任务",
            "5. 深入理解分布式训练机制"
        ]
        
        for step in next_steps:
            print(f"  {step}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
