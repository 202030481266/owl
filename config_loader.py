import json
import os
from pathlib import Path
from dotenv import load_dotenv


class ConfigLoader:
    _config = None

    @classmethod
    def load_config(cls, config_path="config.json"):
        """加载配置文件并初始化环境变量"""
        if cls._config is None:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cls._config = json.load(f)

                # 加载 .env 文件
                env_path = cls._config["paths"].get("env_file", ".env")
                env_path = Path(__file__).parent / env_path
                if env_path.exists():
                    load_dotenv(dotenv_path=env_path)
                else:
                    raise FileNotFoundError(f".env file not found at {env_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load config: {str(e)}")
        return cls._config

    @classmethod
    def get(cls, *keys, default=None):
        """获取配置值，支持嵌套键"""
        config = cls._config
        for key in keys:
            config = config.get(key, default)
            if config is default:
                return default
        return config