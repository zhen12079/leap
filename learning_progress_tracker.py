#!/usr/bin/env python3
"""
LeapAI框架学习进度跟踪器

用于跟踪和管理学习进度，提供学习状态检查和下一步建议。
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

class LearningProgressTracker:
    """学习进度跟踪器"""
    
    def __init__(self):
        self.progress_file = Path(__file__).parent / "learning_progress.json"
        self.progress_data = self.load_progress()
        
    def load_progress(self):
        """加载学习进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载进度文件失败: {e}")
        
        # 默认进度数据
        return {
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "stages": {
                "stage1_architecture": {
                    "name": "理解框架整体架构和设计理念",
                    "status": "completed",
                    "completed_time": None,
                    "notes": ""
                },
                "stage2_config_system": {
                    "name": "学习配置系统和入口机制", 
                    "status": "completed",
                    "completed_time": None,
                    "notes": ""
                },
                "stage3_data_module": {
                    "name": "深入理解数据模块和数据处理流程",
                    "status": "completed",
                    "completed_time": None,
                    "notes": ""
                },
                "stage4_model_building": {
                    "name": "学习模型构建和NodeGraph机制",
                    "status": "completed",
                    "completed_time": None,
                    "notes": ""
                },
                "stage5_multitask_training": {
                    "name": "理解多任务训练和拓扑定义",
                    "status": "completed",
                    "completed_time": None,
                    "notes": ""
                },
                "stage6_perception_tasks": {
                    "name": "学习感知任务的具体实现",
                    "status": "pending",
                    "completed_time": None,
                    "notes": ""
                },
                "stage7_distributed_training": {
                    "name": "掌握分布式训练和部署机制",
                    "status": "pending",
                    "completed_time": None,
                    "notes": ""
                },
                "stage8_practice_training": {
                    "name": "实践：运行一个完整的训练任务",
                    "status": "pending",
                    "completed_time": None,
                    "notes": ""
                },
                "stage9_practice_add_task": {
                    "name": "实践：添加一个新的感知任务",
                    "status": "pending",
                    "completed_time": None,
                    "notes": ""
                },
                "stage10_practice_extend": {
                    "name": "实践：修改和扩展现有组件",
                    "status": "pending",
                    "completed_time": None,
                    "notes": ""
                }
            },
            "practice_files": {
                "step1_understanding_architecture.py": "completed",
                "step2_config_system_practice.py": "completed", 
                "step3_data_module_practice.py": "completed",
                "step4_model_building_practice.py": "completed",
                "step5_multitask_practice.py": "completed",
                "step6_perception_tasks_practice.py": "pending",
                "step7_distributed_practice.py": "pending",
                "run_training_demo.py": "pending"
            },
            "total_time_spent": 0,
            "current_focus": "stage6_perception_tasks"
        }
    
    def save_progress(self):
        """保存学习进度"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存进度文件失败: {e}")
    
    def update_stage_status(self, stage_key, status, notes=""):
        """更新阶段状态"""
        if stage_key in self.progress_data["stages"]:
            old_status = self.progress_data["stages"][stage_key]["status"]
            self.progress_data["stages"][stage_key]["status"] = status
            self.progress_data["stages"][stage_key]["notes"] = notes
            
            if status == "completed" and old_status != "completed":
                self.progress_data["stages"][stage_key]["completed_time"] = datetime.now().isoformat()
                print(f"🎉 阶段完成: {self.progress_data['stages'][stage_key]['name']}")
            
            self.save_progress()
            return True
        return False
    
    def update_practice_file_status(self, filename, status):
        """更新练习文件状态"""
        if filename in self.progress_data["practice_files"]:
            self.progress_data["practice_files"][filename] = status
            self.save_progress()
            return True
        return False
    
    def get_current_stage(self):
        """获取当前学习阶段"""
        for stage_key, stage_data in self.progress_data["stages"].items():
            if stage_data["status"] == "in_progress":
                return stage_key, stage_data
        return None, None
    
    def get_next_stage(self):
        """获取下一个待学习阶段"""
        for stage_key, stage_data in self.progress_data["stages"].items():
            if stage_data["status"] == "pending":
                return stage_key, stage_data
        return None, None
    
    def show_progress_summary(self):
        """显示进度摘要"""
        print("=" * 60)
        print("📊 LeapAI框架学习进度摘要")
        print("=" * 60)
        
        completed = sum(1 for stage in self.progress_data["stages"].values() if stage["status"] == "completed")
        in_progress = sum(1 for stage in self.progress_data["stages"].values() if stage["status"] == "in_progress")
        pending = sum(1 for stage in self.progress_data["stages"].values() if stage["status"] == "pending")
        total = len(self.progress_data["stages"])
        
        print(f"总进度: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"已完成: {completed} | 进行中: {in_progress} | 待开始: {pending}")
        
        print("\n📋 各阶段状态:")
        for i, (stage_key, stage_data) in enumerate(self.progress_data["stages"].items(), 1):
            status_icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}[stage_data["status"]]
            print(f"  {i:2d}. {status_icon} {stage_data['name']}")
        
        print("\n📝 练习文件状态:")
        for filename, status in self.progress_data["practice_files"].items():
            status_icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}[status]
            print(f"  {status_icon} {filename}")
        
        current_stage_key, current_stage = self.get_current_stage()
        if current_stage:
            print(f"\n🎯 当前重点: {current_stage['name']}")
        
        next_stage_key, next_stage = self.get_next_stage()
        if next_stage:
            print(f"🔜 下一步: {next_stage['name']}")
    
    def show_next_steps(self):
        """显示下一步建议"""
        print("\n" + "=" * 60)
        print("🎯 下一步学习建议")
        print("=" * 60)
        
        current_stage_key, current_stage = self.get_current_stage()
        
        if current_stage_key == "stage3_data_module":
            print("📊 当前阶段: 深入理解数据模块和数据处理流程")
            print("\n📚 建议学习内容:")
            print("  1. 运行 step3_data_module_practice.py 练习脚本")
            print("  2. 理解多任务数据加载机制")
            print("  3. 学习数据预处理和增强流程")
            print("  4. 掌握目标生成和标签处理")
            
            print("\n🔧 关键文件:")
            print("  • leapai/data/data_module.py - 数据模块核心")
            print("  • leapai/data/dataloader/ - 数据加载器")
            print("  • leapai/data/transform/ - 数据变换")
            print("  • leapai/data/target/ - 目标生成")
            
            print("\n✅ 完成标准:")
            print("  • 理解多任务数据加载原理")
            print("  • 能够配置和使用数据模块")
            print("  • 掌握数据预处理流程")
            print("  • 完成step3练习脚本")
        
        elif current_stage_key == "stage4_model_building":
            print("🏗️ 当前阶段: 学习模型构建和NodeGraph机制")
            print("\n📚 建议学习内容:")
            print("  1. 理解NodeGraph设计理念")
            print("  2. 学习节点化模型构建")
            print("  3. 掌握模型拓扑定义")
            print("  4. 实践模型组件开发")
        
        else:
            print("📖 继续按照学习指南进行学习")
    
    def mark_stage_complete(self, stage_key):
        """标记阶段完成"""
        return self.update_stage_status(stage_key, "completed")
    
    def start_stage(self, stage_key):
        """开始新阶段"""
        # 先将当前进行中的阶段设为pending
        for key, stage_data in self.progress_data["stages"].items():
            if stage_data["status"] == "in_progress":
                self.update_stage_status(key, "pending")
        
        # 开始新阶段
        self.update_stage_status(stage_key, "in_progress")
        self.progress_data["current_focus"] = stage_key

def main():
    """主函数 - 显示学习进度和下一步建议"""
    tracker = LearningProgressTracker()
    
    print("🎓 LeapAI框架学习进度跟踪器")
    print("=" * 60)
    
    # 显示进度摘要
    tracker.show_progress_summary()
    
    # 显示下一步建议
    tracker.show_next_steps()
    
    print("\n" + "=" * 60)
    print("💡 使用提示:")
    print("  • 运行 python learning_progress_tracker.py 查看进度")
    print("  • 按照学习指南逐步完成各阶段")
    print("  • 完成练习后更新进度状态")
    print("=" * 60)

if __name__ == "__main__":
    main()
