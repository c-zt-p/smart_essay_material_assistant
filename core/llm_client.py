# core/llm_client.py
from typing import AsyncGenerator, List, Dict, Any, Union
from magentic import prompt, SystemMessage, UserMessage, AssistantMessage, AnyMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic import AsyncStreamedStr

from .settings_manager import settings, ModelSetting
# 假设 MultiModalChunkOutput 和相关类型在你项目的某个地方定义
# from ..primitives.chunk import MultiModalChunkOutput, MultiModalChunkType # 如果适用

class LLMClient:
    def __init__(self, model_name: str = None): # model_name 来自UI选择器或默认
        self.model_config: ModelSetting = settings.get_chat_model(model_name)
        if not self.model_config:
            raise ValueError(f"无法找到名为 {model_name or '默认'} 的聊天模型配置")

        # 在 __init__ 中创建并存储模型实例，供后续的 @prompt 使用
        self.chat_model_instance = OpenaiChatModel(
            model=self.model_config.model_name,
            api_key=self.model_config.api_key,
            base_url=self.model_config.model_url,
            max_tokens=self.model_config.token_limit
            # temperature=0.7 # Magentic 的 OpenaiChatModel 也接受 temperature
        )
        print(f"LLMClient initialized with model: {self.model_config.name} ({self.model_config.model_name}) at {self.model_config.model_url}")

    async def _prepare_messages(self, query: str, history: List[Dict[str, str]] = None) -> List[AnyMessage]:
        messages: List[AnyMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(UserMessage(msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AssistantMessage(msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(msg["content"]))
        messages.append(UserMessage(query))
        return messages

    async def stream_chat(
        self, query: str, history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]: # 修改返回类型以匹配UI期望 (str chunks)
        """
        Streams chat responses as simple text chunks.
        """
        messages_list = await self._prepare_messages(query, history)

        # --- 在方法内部定义 @prompt 修饰的函数 ---
        # 将 self.chat_model_instance 作为 model 参数传递给装饰器
        @prompt(
            "{messages_arg}", # 使用不同的参数名以避免与外部变量 messages_list 混淆
            model=self.chat_model_instance # **关键修改**
        )
        async def _internal_streaming_call(messages_arg: List[AnyMessage]) -> AsyncStreamedStr:
            ...
        # --- 结束内部函数定义 ---
        
        # 现在调用这个内部函数时，不需要再传递 model 参数
        # messages_arg 的值将是 messages_list
        response_stream: AsyncStreamedStr = await _internal_streaming_call(messages_arg=messages_list)

        async for chunk_text in response_stream: # AsyncStreamedStr 直接产生字符串块
            yield chunk_text
            # 如果需要返回 MultiModalChunkOutput:
            # yield MultiModalChunkOutput(
            #     chunk_type=MultiModalChunkType.TEXT, text=chunk_text
            # )

# --- 用于测试的 main (如果需要) ---
async def main_test():
    print("Testing LLMClient...")
    try:
        # 确保 settings.json 中有一个名为 "your-default-chat-model-name" (或默认模型) 的配置
        # 替换为你的 settings.json 中真实存在的模型名称
        default_model_name_from_settings = settings.default_models.get("chat_model")
        if not default_model_name_from_settings:
            if settings.chat_models:
                default_model_name_from_settings = settings.chat_models[0].name
            else:
                print("错误：settings.json 中没有找到聊天模型配置。")
                return
        
        print(f"Using model for test: {default_model_name_from_settings}")
        client = LLMClient(model_name=default_model_name_from_settings)
        
        test_query = "你好，请介绍一下你自己。"
        print(f"\nStreaming chat for query: '{test_query}'")
        
        full_response = ""
        async for chunk in client.stream_chat(query=test_query, history=[]):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print("\n\nFull response received:")
        print(full_response)

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 这个 main_test 仅用于独立测试 LLMClient
    # 在实际应用中，LLMClient 会被 ChatWindow 使用
    # import asyncio
    # asyncio.run(main_test())
    pass