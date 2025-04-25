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
    download_in_progress = False
    
    # 解析参数，添加--no-disconnect选项
    wait_for_download = "--no-disconnect" in sys.argv
    if wait_for_download:
        sys.argv.remove("--no-disconnect")
    
    # 解析超时参数
    timeout = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--timeout="):
            timeout = int(arg.split("=")[1])
            sys.argv.pop(i)
            break

    try:
        await mcp_toolkit.connect()

        # Default task
        default_task = (
            "Help me find the Transformer author and download the paper."
        )

        # Override default task if command line argument is provided
        task = sys.argv[1] if len(sys.argv) > 1 else default_task
        
        # 检测是否包含下载相关的关键词
        download_in_progress = "download" in task.lower()

        # Connect to all MCP toolkits
        tools = [*mcp_toolkit.get_tools()]
        society = await construct_society(task, tools)
        answer, chat_history, token_count = await arun_society(society)
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=4)
        print(f"\033[94mAnswer: {answer}\033[0m")
        
        # 如果检测到下载操作且需要等待
        if download_in_progress and wait_for_download:
            print("检测到下载操作，等待下载完成...")
            if timeout:
                print(f"将在 {timeout} 秒后自动断开连接")
                try:
                    # 等待用户输入或超时
                    await asyncio.wait_for(
                        asyncio.create_task(wait_for_user_input("下载完成后按回车键断开连接...")), 
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    print(f"等待超时 ({timeout}秒)，准备断开连接")
            else:
                # 无超时，等待用户确认
                await wait_for_user_input("下载完成后按回车键断开连接...")

    finally:
        # Make sure to disconnect safely after all operations are completed.
        try:
            print("准备断开与MCP服务器的连接...")
            await mcp_toolkit.disconnect()
            print("已成功断开连接")
        except Exception as e:
            print(f"断开连接失败: {str(e)}")


async def wait_for_user_input(prompt: str) -> None:
    """等待用户输入"""
    print(prompt)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sys.stdin.readline)


if __name__ == "__main__":
    asyncio.run(main())
