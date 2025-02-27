import csv
import hashlib
from dsClient import *

class knowledgeTransformer(dsClient):
    def __init__(self, client, retriever, 
                monitor_dir="./chatHistory", 
                export_dir="./knowledgeBase",
                personality="知识转换器"):
        super().__init__(client, retriever, personality)
        self.monitor_dir = monitor_dir        # 监控的历史记录文件夹
        self.export_dir = export_dir          # 导出的知识文件夹
        self.processed_files = set()           # 已处理文件记录
        self._init_dirs()

    def _init_dirs(self):
        """初始化必要目录"""
        os.makedirs(self.monitor_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def _build_system_prompt(self, knowledge):
        """构建知识提取专用提示词"""
        return f"""你是一个专业的聊天对话总结人员，请从对话聊天历史记录中提取结构化知识。例如从对话中总结用户的爱好是什么，今天经历了糟糕的工作等：
1. 根据聊天长度生成合适个数重要的内容,最多10个
2. 每个知识点包含[主题、具体内容、相关对话时间戳]
3. 用中文输出，格式为：
   - 主题: <简短标题>
   描述: <详细说明, 最多30字>
   来源: <相关时间戳>

【原始对话记录】
{knowledge}

"""

    def _get_file_hash(self, file_path):
        """计算文件哈希值用于内容校验"""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def detect_unprocessed_files(self):
        """检测需要处理的文件（返回待处理文件列表）"""
        processed = {
            os.path.splitext(f)[0] 
            for f in os.listdir(self.export_dir) 
            if f.endswith(".csv")
        }
        
        unprocessed = []
        for f in os.listdir(self.monitor_dir):
            if f.endswith(".json"):
                base_name = os.path.splitext(f)[0]
                csv_path = os.path.join(self.export_dir, f"{base_name}.csv")
                
                # 检查是否已处理且内容未修改
                if not os.path.exists(csv_path):
                    unprocessed.append(f)
                else:
                    json_hash = self._get_file_hash(os.path.join(self.monitor_dir, f))
                    csv_hash = self._get_file_hash(csv_path)
                    if json_hash != csv_hash:
                        unprocessed.append(f)
        return unprocessed

    def _convert_json_to_text(self, json_path):
        """将JSON对话记录转换为文本"""
        with open(json_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        
        return "\n".join(
            f"[{conv['timestamp']}] 用户: {conv['user']}\n助手: {conv['assistant']}"
            for conv in conversations
        )

    def _parse_knowledge(self, raw_text):
        """解析模型返回的知识点"""
        knowledge_points = []
        current_point = {}
        
        for line in raw_text.split("\n"):
            if line.startswith("- 主题:"):
                if current_point:
                    knowledge_points.append(current_point)
                    current_point = {}
                current_point["主题"] = line.split(":", 1)[1].strip()
            elif line.startswith("描述:"):
                current_point["描述"] = line.split(":", 1)[1].strip()
            elif line.startswith("来源:"):
                current_point["来源"] = line.split(":", 1)[1].strip()
        
        if current_point:
            knowledge_points.append(current_point)
        return knowledge_points

    def convert_to_knowledge(self, batch_size=5):
        """批量转换未处理的对话记录"""
        unprocessed = self.detect_unprocessed_files()
        if not unprocessed:
            print("没有需要处理的文件")
            return

        for i, json_file in enumerate(unprocessed[:batch_size]):
            try:
                # 读取原始对话记录
                json_path = os.path.join(self.monitor_dir, json_file)
                conversation_text = self._convert_json_to_text(json_path)
                
                # 调用模型生成知识
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self._build_system_prompt(conversation_text)},
                        {"role": "user", "content": "请提取关键知识点"}
                    ],
                    temperature=0.3,
                    max_tokens=1024
                )
                
                # 解析并保存知识
                knowledge = self._parse_knowledge(response.choices[0].message.content)
                csv_file = os.path.splitext(json_file)[0] + ".csv"
                self._save_to_csv(knowledge, csv_file)
                
                print(f"已处理 {json_file} -> {csv_file} ({i+1}/{len(unprocessed)})")
                
            except Exception as e:
                print(f"处理 {json_file} 失败: {str(e)}")

    def _save_to_csv(self, knowledge, filename):
        """保存知识点到CSV文件"""
        csv_path = os.path.join(self.export_dir, filename)
        
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["主题", "描述", "来源"])
            writer.writeheader()
            writer.writerows(knowledge)
