import asyncio
import sys
import json
import os
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType
from camel.logger import set_log_level, get_logger
from camel.toolkits import MCPToolkit

from owl.utils.enhanced_role_playing import OwlRolePlaying, arun_society

import logging
import requests
import time
import glob
import zipfile
import io

# 设置环境变量
basedir = Path(__file__).parent
env_path = basedir / ".env"
mcp_config_path = basedir / "mcp_servers_config.json"
load_dotenv(dotenv_path=env_path)

# 设置日志
set_log_level(level="DEBUG")
logger = get_logger(__name__)
file_handler = logging.FileHandler("PaperAgent.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

root_logger = logging.getLogger()
root_logger.addHandler(file_handler)


# 创建roleplay模型
def create_role_play_model(model_platform: str, model_type: Optional[str] = None):
    if model_platform not in ['siliconflow', 'volcano']:
        raise ValueError(f"Invalid model platform: {model_platform}")
    if model_platform == 'siliconflow':
        model_type = model_type or "deepseek-ai/DeepSeek-V3"
        return ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=model_type,
            model_config_dict={"temperature": 0.6, "max_tokens": 16384},
        )
    else:
        model_type = model_type or "ep-20250326192434-95gld" # deepseek-V3
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            url=os.getenv("ARK_API_BASE_URL"),
            api_key= os.getenv("ARK_API_KEY"),
            model_config_dict={"temperature": 0.6, "max_tokens": 16384},
        )


# 分析聊天记录
def analyze_chat_history(chat_history, key: str):
    """Analyze chat history and extract tool call information."""
    print(f"\n============ {key} Tool Call Analysis ============")
    logger.info(f"========== Starting {key} tool call analysis ==========")

    tool_calls = []
    for i, message in enumerate(chat_history):
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message.get("tool_calls", []):
                if tool_call.get("type") == "function":
                    function = tool_call.get("function", {})
                    tool_info = {
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                        "message_index": i,
                    }
                    tool_calls.append(tool_info)
                    print(
                        f"Tool Call: {function.get('name')} Args: {function.get('arguments')}"
                    )
                    logger.info(
                        f"Tool Call: {function.get('name')} Args: {function.get('arguments')}"
                    )

        elif message.get("role") == "tool" and "tool_call_id" in message:
            for tool_call in tool_calls:
                if tool_call.get("call_id") == message.get("tool_call_id"):
                    result = message.get("content", "")
                    result_summary = (
                        result[:100] + "..." if len(result) > 100 else result
                    )
                    print(
                        f"Tool Result: {tool_call.get('name')} Return: {result_summary}"
                    )
                    logger.info(
                        f"Tool Result: {tool_call.get('name')} Return: {result_summary}"
                    )

    print(f"Total tool calls found: {len(tool_calls)}")
    logger.info(f"Total tool calls found: {len(tool_calls)}")
    logger.info("========== Finished tool call analysis ==========")

    with open(f"PaperAgent_{key}_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

    print(f"Records saved to PaperAgent_{key}_history.json")
    print(f"============ Analysis Complete ============\n")


# 构建society
async def construct_society(
    question: str,
    tools: List[FunctionTool],
) -> OwlRolePlaying:
    r"""build a multi-agent OwlRolePlaying instance.

    Args:
        question (str): The main theme of papers to be downloaded.
        tools (List[FunctionTool]): The MCP tools to use.
    """
    models = {
        "user": create_role_play_model(
            model_platform="volcano",
        ),
        "assistant": create_role_play_model(
            model_platform="volcano",
        ),
    }

    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {
        "model": models["assistant"],
        "tools": tools,
    }

    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False, # ^ 如果任务和模糊，可以使用task_specify
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    return society


# 读取pdf文件的内容
# https://mineru.net/apiManage
def read_pdf_content(pdf_path: str, is_directory: bool = False, output_dir: str = "./output") -> dict:
    """
    使用mineru的API解析PDF文件内容，并将结果ZIP文件保存到指定目录
    
    Args:
        pdf_path (str): PDF文件的路径或包含PDF文件的文件夹路径
        is_directory (bool): 是否为文件夹路径，如为True则处理文件夹中所有PDF文件
        output_dir (str): 保存结果ZIP文件的目录路径
        
    Returns:
        dict: 处理结果，格式为 {文件名: 状态信息} 的字典
    """
    # 从环境变量获取API密钥
    api_key = os.getenv("MINERU_API_KEY")
    if not api_key:
        logger.error("未找到MINERU_API_KEY环境变量")
        return {"error": "未找到MINERU_API_KEY环境变量"}
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集需要处理的PDF文件
    pdf_files = []
    file_names = []
    
    if is_directory:
        # 检查目录是否存在
        if not os.path.isdir(pdf_path):
            logger.error(f"目录不存在: {pdf_path}")
            return {"error": f"目录不存在: {pdf_path}"}
            
        # 获取目录中所有PDF文件
        pdf_pattern = os.path.join(pdf_path, "*.pdf")
        for pdf_file in glob.glob(pdf_pattern):
            pdf_files.append(pdf_file)
            file_names.append(os.path.basename(pdf_file))
            
        if not pdf_files:
            logger.error(f"目录中未找到PDF文件: {pdf_path}")
            return {"error": f"目录中未找到PDF文件: {pdf_path}"}
    else:
        # 检查单个文件是否存在
        if not os.path.exists(pdf_path):
            logger.error(f"文件不存在: {pdf_path}")
            return {"error": f"文件不存在: {pdf_path}"}
            
        pdf_files.append(pdf_path)
        file_names.append(os.path.basename(pdf_path))
    
    # 文件批量上传解析
    upload_url = 'https://mineru.net/api/v4/file-urls/batch'
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备批量上传请求数据
    files_data = []
    for i, file_name in enumerate(file_names):
        files_data.append({
            "name": file_name, 
            "is_ocr": False, 
            "data_id": file_name
        })
        
    data = {
        "enable_formula": True,
        "language": "auto", # * English by default
        "layout_model": "doclayout_yolo",
        "enable_table": False,
        "files": files_data
    }
    
    try:
        # 请求上传URL
        response = requests.post(upload_url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"请求上传URL失败: {response.status_code}, {response.text}")
            return {"error": f"请求上传URL失败: {response.status_code}"}
            
        result = response.json()
        if result.get("code") != 0:
            error_msg = result.get("msg", "未知错误")
            logger.error(f"申请上传URL失败: {error_msg}")
            return {"error": f"申请上传URL失败: {error_msg}"}
            
        # 根据API文档，正确的字段是data.batch_id和data.file_urls
        batch_id = result["data"]["batch_id"]
        # 确保file_urls字段存在
        if "file_urls" not in result["data"]:
            logger.error(f"响应中缺少file_urls字段: {result}")
            return {"error": "响应格式不符合预期，缺少file_urls字段"}
            
        upload_urls = result["data"]["file_urls"]
        
        # 上传所有文件
        for i, (pdf_file, upload_url) in enumerate(zip(pdf_files, upload_urls)):
            with open(pdf_file, 'rb') as f:
                res_upload = requests.put(upload_url, data=f)
                if res_upload.status_code != 200:
                    logger.error(f"文件 {file_names[i]} 上传失败: {res_upload.status_code}")
                    # 继续上传其他文件
                else:
                    logger.info(f"文件 {file_names[i]} 上传成功")
                    
        logger.info(f"所有文件上传完成，批处理ID: {batch_id}")
        
        # 等待处理完成并获取结果
        max_attempts = 100 * len(pdf_files)  # 增加最大尝试次数，因为处理多个文件需要更多时间
        attempt = 0
        result_dict = {}
        processed_files = set()
        
        while attempt < max_attempts and len(processed_files) < len(pdf_files):
            attempt += 1
            
            # 查询进度
            result_url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
            res = requests.get(result_url, headers=headers)
            
            if res.status_code != 200:
                logger.error(f"获取结果失败: {res.status_code}, {res.text}")
                if len(result_dict) > 0:
                    # 返回已处理的结果
                    return result_dict
                return {"error": f"获取结果失败: {res.status_code}"}
                
            result_data = res.json()
            
            # 检查响应格式
            if result_data.get("code") != 0:
                logger.error(f"获取结果失败: {result_data.get('msg', '未知错误')}")
                continue
                
            # 根据API文档，响应中应该有extract_result字段
            extract_result = result_data.get("data", {}).get("extract_result", [])
            if not extract_result:
                logger.warning("响应中没有extract_result字段或为空")
                time.sleep(3)
                continue
                
            # 处理已完成的任务
            all_done = True
            for task in extract_result:
                if not isinstance(task, dict):
                    logger.error(f"收到意外的响应格式，task不是字典: {task}")
                    continue
                    
                file_name = task.get("file_name", "")
                state = task.get("state", "")
                
                if not file_name or file_name not in file_names:
                    continue
                    
                if file_name in processed_files:
                    continue
                    
                # 根据API文档，状态应为done、running、pending、failed等
                if state == "done":
                    # 获取解析结果
                    zip_url = task.get("full_zip_url", "")
                    if zip_url:
                        try:
                            # 下载并保存ZIP文件
                            zip_response = requests.get(zip_url)
                            if zip_response.status_code == 200:
                                # 生成输出文件名（去掉.pdf后缀，添加.zip后缀）
                                base_name = os.path.splitext(file_name)[0]
                                output_file = os.path.join(output_dir, f"{base_name}.zip")
                                
                                # 保存ZIP文件
                                with open(output_file, 'wb') as f:
                                    f.write(zip_response.content)
                                
                                # 提取里面的full.md文件内容，这是文本内容，给大模型处理
                                try:
                                    # 从ZIP中提取full.md内容
                                    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zf:
                                        # 查找full.md文件
                                        md_files = [f for f in zf.namelist() if f.endswith('full.md')]
                                        if md_files:
                                            with zf.open(md_files[0]) as md_file:
                                                md_content = md_file.read().decode('utf-8')
                                                # 返回文件内容而不是成功信息
                                                result_dict[file_name] = md_content
                                                logger.info(f"已从文件 {file_name} 中提取full.md内容，共{len(md_content)}字符")
                                        else:
                                            logger.warning(f"ZIP文件中未找到full.md: {output_file}")
                                            # 保存ZIP但不提取内容
                                            result_dict[file_name] = f"处理成功但未找到full.md文件: {output_file}"
                                    processed_files.add(file_name)
                                except Exception as ex:
                                    logger.error(f"提取full.md内容时发生错误: {str(ex)}")
                                    result_dict[file_name] = f"提取full.md内容时发生错误: {str(ex)}"
                                    processed_files.add(file_name)
                            else:
                                logger.error(f"下载解析结果ZIP失败: {zip_response.status_code}")
                                result_dict[file_name] = f"下载解析结果失败: {zip_response.status_code}"
                                processed_files.add(file_name)
                        except Exception as e:
                            logger.error(f"下载ZIP文件时出错: {str(e)}")
                            result_dict[file_name] = f"下载ZIP文件时出错: {str(e)}"
                            processed_files.add(file_name)
                    else:
                        result_dict[file_name] = "解析完成但未返回下载链接"
                        processed_files.add(file_name)
                        logger.warning(f"文件 {file_name} 解析完成但未返回下载链接")
                elif state == "failed":
                    error_message = task.get("err_msg", "未知错误")
                    result_dict[file_name] = f"处理失败: {error_message}"
                    processed_files.add(file_name)
                    logger.error(f"文件 {file_name} 处理失败: {error_message}")
                else:
                    # 任务仍在处理中
                    all_done = False
                    progress_info = task.get("extract_progress", {})
                    if progress_info:
                        extracted = progress_info.get("extracted_pages", 0)
                        total = progress_info.get("total_pages", 0)
                        if total > 0:
                            logger.info(f"文件 {file_name} 处理进度: {extracted}/{total} 页")
            
            # 如果所有文件都已处理完成，提前退出
            if len(processed_files) >= len(pdf_files) or all_done:
                break
                
            # 等待后再次查询
            time.sleep(3)
        
        # 添加未处理完成的文件
        for file_name in file_names:
            if file_name not in processed_files:
                result_dict[file_name] = "处理超时或未返回结果"
                logger.warning(f"文件 {file_name} 处理超时或未返回结果")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"PDF批量解析过程中发生错误: {str(e)}")
        return {"error": f"PDF批量解析过程中发生错误: {str(e)}"}

# 简单测试mineru的API
def test_read_pdf_content():
    pdf_folder = "./test"
    output_dir = "./test/output"
    result = read_pdf_content(pdf_folder, True, output_dir)
    if "error" in result:
        print(result["error"])
    else:
        with open("./test/result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


# 总结论文内容，使用思考模型
def summarize_paper(paper_content: str) -> str:
    """
    总结论文内容，使用思考模型
    
    Args:
        paper_content (str): 论文内容
        
    Returns:
        str: 总结内容
    """
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url=os.environ.get("ARK_API_BASE_URL"),
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=os.environ.get("ARK_API_KEY"),
    )
    
    summary_prompt = """You are an excellent academic paper reviewer. You conduct paper summarization on the full paper text provided by the user, with following instructions:

REVIEW INSTRUCTION:

Summary of Academic Paper's Technical Approach

Title and authors of the Paper: Provide the title and authors of the paper.

Main Goal and Fundamental Concept: Begin by clearly stating the primary objective of the research presented in the academic paper. Describe the core idea or hypothesis that underpins the study in simple, accessible language.

Technical Approach: Provide a detailed explanation of the methodology used in the research. Focus on describing how the study was conducted, including any specific techniques, models, or algorithms employed. Avoid delving into complex jargon or highly technical details that might obscure understanding.

Distinctive Features: Identify and elaborate on what sets this research apart from other studies in the same field. Highlight any novel techniques, unique applications, or innovative methodologies that contribute to its distinctiveness.

Experimental Setup and Results: Describe the experimental design and data collection process used in the study. Summarize the results obtained or key findings, emphasizing any significant outcomes or discoveries.

Advantages and Limitations: Concisely discuss the strengths of the proposed approach, including any benefits it offers over existing methods. Also, address its limitations or potential drawbacks, providing a balanced view of its efficacy and applicability.

Conclusion: Sum up the key points made about the paper's technical approach, its uniqueness, and its comparative advantages and limitations. Aim for clarity and succinctness in your summary.

OUTPUT INSTRUCTIONS:

Only use the headers provided in the instructions above.
Format your output in clear, human-readable Markdown.
Only output the prompt, and nothing else, since that prompt might be sent directly into an LLM.
PAPER TEXT INPUT: {}
"""
    response = client.chat.completions.create(
        model="doubao-1-5-thinking-pro-250415",
        messages=[
            {
                "role": "user",
                "content": summary_prompt.format(paper_content)
            }
        ],
    )
    return response.choices[0].message.content


# workflow + multi-agent
async def main(step='all', project_dir='D:\\项目\\dev\\python\\owl\\papers'):
    mcp_toolkit = MCPToolkit(config_path=str(mcp_config_path))
    
    # * workflow input variables, remember that change the mcp servers config too.
    PAPER_DATA_DIR = project_dir + "\\data"
    PAPER_ANALYSIS_DIR = project_dir + "\\analysis"
    
    print(f"\033[94m执行步骤: {step}\033[0m")
    logger.info(f"执行步骤: {step}")
    
    try:
        await mcp_toolkit.connect()
        
        # 确保必要的目录存在
        os.makedirs(PAPER_DATA_DIR, exist_ok=True)
        os.makedirs(PAPER_ANALYSIS_DIR, exist_ok=True)
        
        # 下载论文步骤
        if step in ['download', 'all']:
            search_task = (
                "你是一位专业的研究助手，请你帮我搜索并下载10篇最近（2022-2025年）关于解决Transformer模型上下文长度限制的前沿研究论文。\n\n"
                "请特别关注以下几个方向的论文：\n"
                "1. 长序列建模技术：如StreamingLLM、Transformer-XL、Longformer、xPos等改进Transformer架构的方法\n"
                "2. 注意力机制优化：稀疏注意力、分组注意力、线性复杂度注意力等降低计算复杂度的方法\n"
                "3. 记忆增强方法：外部记忆机制、检索增强、压缩记忆等扩展上下文能力的技术\n"
                "4. 位置编码创新：RoPE、ALiBi、xPos等适应长序列的位置编码方案\n"
                "5. 递归状态传递：State Space Models、Mamba等具有递归记忆能力的模型\n\n"
                "请执行以下步骤：\n"
                "1. 使用搜索工具查找相关论文，确保覆盖上述五个关键方向\n"
                f"2. 对于每篇论文，下载PDF文件到本地'{PAPER_DATA_DIR}'目录\n"
                "3. 为每篇论文创建一个合理的文件名，包含年份和关键主题\n"
                "4. 确保下载的论文质量高、来源可靠（如arXiv、顶会论文等）\n"
                f"5. 创建一个索引文件'{project_dir}\\paper_index.json'，包含所有下载论文的元数据（标题、作者、年份、文件路径、主要方向）\n\n"
                "请确保下载的论文具有多样性，覆盖不同的技术路线和方法。"
            )
            
            # Connect to all MCP toolkits
            tools = [*mcp_toolkit.get_tools()]
            
            # 第一步：下载论文
            print("\033[94m开始下载论文...\033[0m")
            logger.info("开始下载论文")
            society = await construct_society(search_task, tools)
            download_answer, download_history, download_token_count = await arun_society(society)
            print(f"\033[94m论文下载完成: {download_answer}\033[0m")
            print(f"\033[93m下载阶段token使用情况: 提示词tokens: {download_token_count['prompt_token_count']}, 生成tokens: {download_token_count['completion_token_count']}\033[0m")
            logger.info(f"下载阶段token使用情况: 提示词tokens: {download_token_count['prompt_token_count']}, 生成tokens: {download_token_count['completion_token_count']}")
            
            # 保存聊天历史用于分析
            analyze_chat_history(download_history, "download")
            
            # 如果只执行下载步骤，则退出
            if step == 'download':
                print("\033[94m下载步骤完成，退出程序\033[0m")
                return
        
        # 分析论文步骤
        if step in ['analyze', 'all']:
            print("\033[94m开始分析论文...\033[0m")
            logger.info("开始分析论文")
            
            # 直接从PAPER_DATA_DIR读取PDF文件
            pdf_files = glob.glob(os.path.join(PAPER_DATA_DIR, "*.pdf"))
            
            # 检查是否找到PDF文件
            if not pdf_files:
                logger.error(f"在 {PAPER_DATA_DIR} 目录中未找到PDF文件")
                print(f"\033[91m在 {PAPER_DATA_DIR} 目录中未找到PDF文件\033[0m")
                return
                
            print(f"\033[94m在 {PAPER_DATA_DIR} 目录中找到 {len(pdf_files)} 个PDF文件\033[0m")
            logger.info(f"在 {PAPER_DATA_DIR} 目录中找到 {len(pdf_files)} 个PDF文件")
            
            # 使用异步协程处理论文分析
            async def process_paper(pdf_path):
                """异步处理单篇论文的函数"""
                try:
                    # 从文件名生成论文标题
                    base_name = os.path.basename(pdf_path)
                    paper_title = os.path.splitext(base_name)[0].replace("_", " ").title()
                    
                    # 生成输出文件名
                    output_filename = os.path.basename(pdf_path).replace(".pdf", ".md")
                    output_path = os.path.join(PAPER_ANALYSIS_DIR, output_filename)
                    
                    print(f"\033[96m正在处理论文: {paper_title}\033[0m")
                    logger.info(f"正在处理论文: {paper_title}, 路径: {pdf_path}")
                    
                    # 读取PDF内容 (使用run_in_executor将同步函数转为异步)
                    loop = asyncio.get_running_loop()
                    pdf_content_result = await loop.run_in_executor(
                        None, 
                        lambda: read_pdf_content(pdf_path, is_directory=False, output_dir=PAPER_ANALYSIS_DIR)
                    )
                    
                    if "error" in pdf_content_result:
                        logger.error(f"读取PDF内容失败: {pdf_content_result['error']}")
                        return {
                            "title": paper_title,
                            "status": "失败",
                            "error": pdf_content_result["error"]
                        }
                    
                    # 获取PDF内容
                    paper_content = next(iter(pdf_content_result.values()))
                    
                    # 使用大模型总结论文 (同样使用run_in_executor)
                    summary = await loop.run_in_executor(
                        None, 
                        lambda: summarize_paper(paper_content)
                    )
                    
                    # 保存总结内容 (将同步IO操作转换为异步)
                    await loop.run_in_executor(
                        None, 
                        lambda: write_summary_to_file(output_path, paper_title, summary)
                    )
                    
                    logger.info(f"论文 '{paper_title}' 总结已保存到 {output_path}")
                    
                    return {
                        "title": paper_title,
                        "status": "成功",
                        "output_path": output_path
                    }
                except Exception as e:
                    logger.error(f"处理论文异常: {str(e)}")
                    return {
                        "title": os.path.basename(pdf_path),
                        "status": "失败",
                        "error": str(e)
                    }

            # 辅助函数，用于写入文件
            def write_summary_to_file(path, title, content):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {title} - 论文总结\n\n")
                    f.write(content)

            # 使用信号量限制并发量
            semaphore = asyncio.Semaphore(3)  # 最多同时处理3篇论文

            async def process_with_semaphore(pdf_path):
                """使用信号量限制并发处理数量"""
                async with semaphore:
                    return await process_paper(pdf_path)

            # 创建所有论文的处理任务
            tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 过滤出正常结果和异常
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"处理论文异常: {str(result)}")
                    print(f"\033[91m处理论文异常: {str(result)}\033[0m")
                else:
                    filtered_results.append(result)
                    if result["status"] == "成功":
                        print(f"\033[92m论文 '{result['title']}' 总结成功\033[0m")
                    else:
                        print(f"\033[91m论文 '{result['title']}' 总结失败: {result.get('error', '未知错误')}\033[0m")

            # 创建总结报告
            summary_report = {
                "total": len(pdf_files),
                "success": sum(1 for r in filtered_results if r["status"] == "成功"),
                "failed": sum(1 for r in filtered_results if r["status"] == "失败"),
                "papers": filtered_results
            }

            # 保存总结报告
            with open(f"{project_dir}/summary_report.json", "w", encoding="utf-8") as f:
                json.dump(summary_report, f, ensure_ascii=False, indent=2)

            print(f"\033[94m论文分析完成: 总共 {summary_report['total']} 篇, 成功 {summary_report['success']} 篇, 失败 {summary_report['failed']} 篇\033[0m")
            logger.info(f"论文分析完成: 总共 {summary_report['total']} 篇, 成功 {summary_report['success']} 篇, 失败 {summary_report['failed']} 篇")
            
            # 创建综述报告的文件名
            report_filename = f"{project_dir}/transformer_long_context_survey.md"
            
            # 收集成功总结的论文
            successful_papers = [r for r in filtered_results if r["status"] == "成功"]
            
            if successful_papers:
                # 读取所有总结内容
                summaries = []
                for paper in successful_papers:
                    try:
                        with open(paper["output_path"], "r", encoding="utf-8") as f:
                            summaries.append(f.read())
                    except Exception as e:
                        logger.error(f"读取论文总结失败: {paper['output_path']}, 错误: {str(e)}")
                
                # 创建综述报告
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write("# Transformer长上下文能力研究综述报告\n\n")
                    f.write("## 概述\n\n")
                    f.write(f"本报告汇总了 {len(successful_papers)} 篇关于解决Transformer模型上下文长度限制的研究论文的总结。\n\n")
                    f.write("## 研究方向分类\n\n")
                    f.write("1. 长序列建模技术\n")
                    f.write("2. 注意力机制优化\n")
                    f.write("3. 记忆增强方法\n")
                    f.write("4. 位置编码创新\n")
                    f.write("5. 递归状态传递\n\n")
                    f.write("## 论文总结\n\n")
                    
                    for i, summary in enumerate(summaries):
                        f.write(f"### 论文 {i+1}: {successful_papers[i]['title']}\n\n")
                        f.write(summary)
                        f.write("\n\n---\n\n")
                
                print(f"\033[92m综述报告已成功创建: {report_filename}\033[0m")
                logger.info(f"综述报告已成功创建: {report_filename}")
            else:
                logger.warning("没有成功总结的论文，无法创建综述报告")
                print("\033[93m没有成功总结的论文，无法创建综述报告\033[0m")

            # 如果是从all路径执行到这里，而且下载步骤生成了历史记录，则分析历史记录
            if step == 'all' and 'download_history' in locals():
                analyze_chat_history(download_history, "download")

            print(f"\033[94m所有任务已完成，结果已保存\033[0m")

    finally:
        await asyncio.sleep(1)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            await mcp_toolkit.disconnect()
        except Exception as e:
            print(f"Cleanup error (can be ignored): {e}")


# 创建测试索引文件
def create_test_paper_index(project_dir):
    """
    创建测试用的论文索引文件，用于测试论文分析流程
    
    Args:
        project_dir (str): 项目目录路径
    """
    # 查找项目目录下的PDF文件
    pdf_pattern = os.path.join(project_dir, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    if not pdf_files:
        print(f"\033[91m未在 {project_dir} 及其子目录中找到PDF文件\033[0m")
        return False
        
    # 创建测试索引
    test_index = []
    for pdf_file in pdf_files:
        base_name = os.path.basename(pdf_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # 为每个PDF文件创建一条索引记录
        test_index.append({
            "title": name_without_ext.replace("_", " ").title(),
            "file_path": pdf_file,
            "authors": ["测试作者"],
            "year": "2024",
            "direction": "长序列建模技术"
        })
    
    # 保存测试索引文件
    index_file_path = os.path.join(project_dir, "paper_index.json")
    try:
        with open(index_file_path, "w", encoding="utf-8") as f:
            json.dump(test_index, f, ensure_ascii=False, indent=2)
        
        print(f"\033[92m已创建测试索引文件: {index_file_path}，包含 {len(test_index)} 篇论文信息\033[0m")
        return True
    except Exception as e:
        print(f"\033[91m创建测试索引文件失败: {str(e)}\033[0m")
        return False


if __name__ == "__main__":
    try:
        # 解析命令行参数
        import argparse
        parser = argparse.ArgumentParser(description="论文下载与分析工具")
        parser.add_argument('--step', type=str, choices=['download', 'analyze', 'all'], 
                          default='all', help='指定要执行的步骤: download(下载论文), analyze(分析论文), all(全部执行)')
        parser.add_argument('--create-test-index', action='store_true', 
                          help='创建测试索引文件并退出')
        parser.add_argument('--project-dir', type=str, default='D:\\项目\\dev\\python\\owl\\papers',
                          help='指定项目目录路径')
        args = parser.parse_args()
        
        # 如果只是创建测试索引
        if args.create_test_index:
            print("\033[94m开始创建测试索引文件...\033[0m")
            if create_test_paper_index(args.project_dir):
                print("\033[92m测试索引文件创建成功，退出程序\033[0m")
            else:
                print("\033[91m创建测试索引文件失败\033[0m")
            sys.exit(0)
            
        # 执行主程序
        asyncio.run(main(step=args.step, project_dir=args.project_dir))
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        if sys.platform == 'win32':
            try:
                import asyncio.windows_events
                asyncio.windows_events._overlapped = None
            except (ImportError, AttributeError):
                pass