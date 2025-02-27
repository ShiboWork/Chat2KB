from abc import ABC, abstractmethod
import os
import json
from datetime import datetime

class dsClient(ABC):
    def __init__(self, client, retriever, personality):
        self.client = client
        self.retriever = retriever
        self.chat_history = []
        # 历史记录类
        self.conversation_log = []
        self.max_log_entries = 1000
        self.history_dir = "./chatHistory"
        self._bRecordFlag = False

    @abstractmethod
    def _build_system_prompt(self, knowledge):
        """抽象方法：构建系统提示（需子类实现）"""
        pass

    def _format_history(self):
        """格式化最近对话历史"""
        return "\n".join(self.chat_history[-6:]) if self.chat_history else "无"

    def generate_response(self, query):
        """生成响应（通用实现）"""
        try:
            # 知识检索
            docs = self.retriever.invoke(query)
            knowledge = "\n".join(d.page_content for d in docs[:3])

            # 构建系统提示
            system_content = self._build_system_prompt(knowledge)

            # API调用
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=512,
                stream=False
            )

            # 更新历史记录
            self._update_history(f"用户: {query}", f"助手: {response.choices[0].message.content}")
            if (_bRecordFlag):
                self.record_conversation(query, response.choices[0].message.content)

            return response.choices[0].message.content
        except Exception as e:
            return f"系统错误：{str(e)}"

    def _update_history(self, user_input, assistant_response):
        """维护对话历史"""
        max_rounds = 5
        self.chat_history = (self.chat_history + [user_input, assistant_response])[-max_rounds*2:]

    def record_conversation(self, user_input, assistant_response):
        """记录对话到独立日志"""
        clean_input = user_input.strip()[:500]
        clean_response = assistant_response.strip()[:self.max_log_entries]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_log.append({
            "timestamp": timestamp,
            "user": clean_input,
            "assistant": clean_response
        })
        if len(self.conversation_log) > self.max_log_entries:
            self.conversation_log.pop(0)

    def write_history(self):
        """写入历史记录文件"""
        try:
            os.makedirs(self.history_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.history_dir, f"{timestamp}.json")
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.conversation_log, file, ensure_ascii=False, indent=2)
            print(f"对话历史已保存到：{file_path}")
        except Exception as e:
            print(f"保存对话历史失败：{str(e)}")

    def begin_Record_History(self):
        _bRecordFlag = true

    def end_Record_History(self):
        _bRecordFlag = false
