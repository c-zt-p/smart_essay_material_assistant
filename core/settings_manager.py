import json
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

CONFIG_FILE = Path(__file__).parent.parent / "data" / "settings.json"

class ModelSetting(BaseModel):
    name: str
    model_name: str
    model_url: str
    api_key: str
    token_limit: int

class Settings(BaseModel):
    default_models: Dict[str, str]
    chat_models: List[ModelSetting]
    embedding_models: List[ModelSetting]
    rag_db_path: str = Field(default="./rag_databases")

    def get_chat_model(self, name: Optional[str] = None) -> Optional[ModelSetting]:
        target_name = name or self.default_models.get("chat_model")
        if not target_name:
            return self.chat_models[0] if self.chat_models else None
        for model in self.chat_models:
            if model.name == target_name:
                return model
        return self.chat_models[0] if self.chat_models else None

    def get_embedding_model(self, name: Optional[str] = None) -> Optional[ModelSetting]:
        target_name = name or self.default_models.get("embedding_model")
        if not target_name:
            return self.embedding_models[0] if self.embedding_models else None
        for model in self.embedding_models:
            if model.name == target_name:
                return model
        return self.embedding_models[0] if self.embedding_models else None

    @classmethod
    def load(cls) -> 'Settings':
        if not CONFIG_FILE.exists():
            raise FileNotFoundError(f"配置文件 {CONFIG_FILE} 未找到。")
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

settings = Settings.load()

# 创建 RAG 数据库路径
Path(settings.rag_db_path).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    print("Loaded settings:")
    print(f"Default chat model: {settings.get_chat_model().name if settings.get_chat_model() else 'N/A'}")
    print(f"Default embedding model: {settings.get_embedding_model().name if settings.get_embedding_model() else 'N/A'}")
    print(f"All chat models: {[m.name for m in settings.chat_models]}")
    print(f"RAG DB Path: {settings.rag_db_path}")
