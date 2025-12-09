#!/usr/bin/env python3
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
