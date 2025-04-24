import asyncio
import sys
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType
from camel.logger import set_log_level
from camel.toolkits import MCPToolkit

from owl.utils.enhanced_role_playing import OwlRolePlaying, arun_society

basedir = Path(__file__).parent.parent
env_path = basedir / ".env"
mcp_config_path = basedir / "mcp_servers_config.json"
load_dotenv(dotenv_path=env_path)
set_log_level(level="DEBUG")

def create_role_play_model(model_platform: str, model_type: Optional[str]):
    if model_platform not in ['siliconflow', 'volcano']:
        raise ValueError(f"Invalid model platform: {model_platform}")
    if model_platform == 'siliconflow':
        model_type = model_type or "deepseek-ai/DeepSeek-V3"
        return ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=model_type,
            model_config_dict={"temperature": 0.6, "max_tokens": 131072},
        )
    elif model_platform == 'volcano':
        model_type = model_type or "ep-20250326192434-95gld"
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            url=os.getenv("ARK_API_BASE_URL"),
            api_key= os.getenv("ARK_API_KEY"),
            model_config_dict={"temperature": 0.6, "max_tokens": 131072},
        )
        

async def construct_society(
    question: str,
    tools: List[FunctionTool],
) -> OwlRolePlaying:
    r"""build a multi-agent OwlRolePlaying instance.

    Args:
        question (str): The question to ask.
        tools (List[FunctionTool]): The MCP tools to use.
    """
    models = {
        "user": create_role_play_model(
            model_platform="volcano",
            model_type="ep-20250326192434-95gld",
        ),
        "assistant": create_role_play_model(
            model_platform="volcano",
            model_type="ep-20250326192434-95gld",
        ),
    }

    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {
        "model": models["assistant"],
        "tools": tools,
    }

    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    return society


async def main():
    mcp_toolkit = MCPToolkit(config_path=str(mcp_config_path))

    try:
        await mcp_toolkit.connect()

        # Default task
        default_task = (
            "Help me find the Transformer author and download the paper."
        )

        # Override default task if command line argument is provided
        task = sys.argv[1] if len(sys.argv) > 1 else default_task

        # Connect to all MCP toolkits
        tools = [*mcp_toolkit.get_tools()]
        society = await construct_society(task, tools)
        answer, chat_history, token_count = await arun_society(society)
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=4)
        print(f"\033[94mAnswer: {answer}\033[0m")

    finally:
        # Make sure to disconnect safely after all operations are completed.
        try:
            await mcp_toolkit.disconnect()
        except Exception:
            print("Disconnect failed")


if __name__ == "__main__":
    asyncio.run(main())
