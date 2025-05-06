import asyncio
import os
import json
import sys
from typing import List, Optional, Literal
from openai import OpenAI
import logging
import glob

from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType
from camel.logger import set_log_level, get_logger
from camel.toolkits import MCPToolkit
from camel.models import BaseModelBackend
from owl.utils.enhanced_role_playing import OwlRolePlaying, arun_society

from config_loader import ConfigLoader
from utils import read_pdf_content_local_single, analyze_chat_history, strip_markdown_fences
from prompts_en import ACADEMIC_PAPER_SUMMARY_PROMPT_EN, PAPER_COMPARISON_SUMMARY_PROMPT_EN
from jinja2 import Template

from web_app import start_server
import threading
import webbrowser

# 加载配置
config = ConfigLoader.load_config()

# 设置日志
set_log_level(level=config["logging"]["level"])
logger = get_logger(__name__)
file_handler = logging.FileHandler(config["logging"]["file"], encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(config["logging"]["format"])
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)


# 创建roleplay模型
def create_role_play_model(
    model_platform: str,
    model_type: Optional[str] = None
) -> BaseModelBackend:
    # 检查 model_platform 是否出现在配置中
    if model_platform not in config["api"]:
        raise ValueError(f"Invalid model platform: {model_platform}")

    platform_config = config["api"][model_platform]
    if model_platform == 'siliconflow':
        model_type = model_type or platform_config["default_model"]
        return ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=model_type,
            model_config_dict=platform_config['model_config']
        )
    elif model_platform == 'volcano':
        model_type = model_type or platform_config["default_model"]
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            url=os.getenv(platform_config["base_url_env"]),
            api_key=os.getenv(platform_config["api_key_env"]),
            model_config_dict=platform_config['model_config']
        )
    raise ValueError("Not supported model provider: %s!" % model_platform)


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
            model_platform=config["model"]["user"]["model_platform"],
            model_type=config["model"]["user"].get("model_type", None)
        ),
        "assistant": create_role_play_model(
            model_platform=config["model"]["assistant"]["model_platform"],
            model_type=config["model"]["assistant"].get("model_type", None)
        ),
    }

    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {
        "model": models["assistant"],
        "tools": tools,
    }

    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False, # 如果任务和模糊，可以使用task_specify
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    return society


def summarize_paper(paper_content: str) -> str:
    """
    总结论文内容，使用思考模型

    Args:
        paper_content (str): 论文内容

    Returns:
        str: 总结内容
    """
    model_platform = config['model']['analyze']['model_platform']
    model_type = config['model']['analyze'].get('model_type', None)
    client = OpenAI(
        base_url=os.environ.get(config["api"][model_platform]["base_url_env"]),
        api_key=os.environ.get(config["api"][model_platform]["api_key_env"]),
    )

    # 构建总结论文的提示词
    summary_prompt = Template(ACADEMIC_PAPER_SUMMARY_PROMPT_EN).render(paper_content=paper_content)

    response = client.chat.completions.create(
        model=model_type or config['api'][model_platform]['default_model'],
        messages=[{"role": "user", "content": summary_prompt}],
    )
    return response.choices[0].message.content


def generate_comprehensive_report(summaries: List[str], paper_titles: List[str]) -> str:
    """
    使用大语言模型生成综合分析报告，分析论文的异同点和对比。

    Args:
        summaries (List[str]): 每篇论文的总结内容
        paper_titles (List[str]): 每篇论文的标题

    Returns:
        str: 综合分析报告的Markdown内容
    """
    model_platform = config['model']['analyze']['model_platform']
    model_type = config['model']['analyze'].get('model_type', None)
    client = OpenAI(
        base_url=os.environ.get(config["api"][model_platform]["base_url_env"]),
        api_key=os.environ.get(config["api"][model_platform]["api_key_env"]),
    )

    # 格式化输入：将每篇论文的标题和总结组合
    formatted_input = ""
    for title, summary in zip(paper_titles, summaries):
        formatted_input += f"### {title}\n\n{summary}\n\n---\n\n"

    comprehensive_prompt = Template(PAPER_COMPARISON_SUMMARY_PROMPT_EN).render(
        page_number=len(summaries),
        page_content=formatted_input
    )
    response = client.chat.completions.create(
        model=model_type or config['api'][model_platform]['default_model'],
        messages=[{"role": "user", "content": comprehensive_prompt}],
    )
    return response.choices[0].message.content


async def main(
    step: Literal['all', 'download', 'analyze'] = 'all',
    project_dir: str = None,
):
    """
    完场根据主题搜索论文，下载，分析，形成综述和可视化
    Args:
        step: 可选择的，all表示全流程，download只执行搜索和下载，analyze只进行分析论文
        project_dir: 操作的文件夹根目录

    Returns: None
    """
    project_dir = project_dir or config["paths"]["project_dir"]
    mcp_toolkit = MCPToolkit(config_path=config["paths"]["mcp_config"])

    PAPER_DATA_DIR = os.path.join(project_dir, config["paths"]["data_dir"])
    PAPER_ANALYSIS_DIR = os.path.join(project_dir, config["paths"]["analysis_dir"])

    print(f"\033[94m执行步骤: {step}\033[0m")
    logger.info(f"执行步骤: {step}")

    try:
        # 测试mcp server是否连接正常
        await mcp_toolkit.connect()

        tools = mcp_toolkit.get_tools()

        # 确保必要的目录存在
        os.makedirs(PAPER_DATA_DIR, exist_ok=True)
        os.makedirs(PAPER_ANALYSIS_DIR, exist_ok=True)

        # 输入的提示词
        if step in ['download', 'all']:
            search_task = (
                "你是一位专业的研究助手，请你帮我搜索并下载5篇最近（2022-2025年）关于解决Transformer模型上下文长度限制的前沿研究论文。\n\n"
                "请特别关注以下几个方向的论文：\n"
                "1. 长序列建模技术：如StreamingLLM、Transformer-XL、Longformer、xPos等改进Transformer架构的方法\n"
                "2. 注意力机制优化：稀疏注意力、分组注意力、线性复杂度注意力等降低计算复杂度的方法\n"
                "3. 记忆增强方法：外部记忆机制、检索增强、压缩记忆等扩展上下文能力的技术\n"
                "4. 位置编码创新：RoPE、ALiBi、xPos等适应长序列的位置编码方案\n"
                "5. 递归状态传递：State Space Models、Mamba等具有递归记忆能力的模型\n\n"
                "请执行以下步骤：\n"
                f"1. 使用搜索工具查找相关论文，确保覆盖上述五个关键方向\n"
                f"2. 对于每篇论文，下载PDF文件到本地'{PAPER_DATA_DIR}'目录\n"
                f"3. 为每篇论文创建一个合理的文件名（不带文件格式后缀），包含年份和关键主题\n"
                f"4. 确保下载的论文质量高、来源可靠（如arXiv、顶会论文等）\n"
                f"5. 创建一个索引文件'{os.path.join(project_dir, config['paths']['index_file'])}'，包含所有下载论文的元数据（标题、作者、年份、文件路径、主要方向）\n\n"
                "请确保下载的论文具有多样性，覆盖不同的技术路线和方法。"
            )

            # Connect to all MCP toolkits
            tools = [*mcp_toolkit.get_tools()]

            # 第一步：搜索并且下载论文
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
            async def process_paper(pdf_path, pdf_content):
                """异步处理单篇论文的函数"""
                try:
                    # 从文件名生成论文标题
                    base_name = os.path.basename(pdf_path)
                    paper_title = os.path.splitext(base_name)[0].replace("_", " ").title()

                    # 生成输出文件名
                    output_filename = os.path.basename(pdf_path).replace(".pdf", ".md")
                    output_path = os.path.join(PAPER_ANALYSIS_DIR, output_filename)

                    print(f"\033[96m正在分析论文: {paper_title}\033[0m")
                    logger.info(f"正在分析论文: {paper_title}, 路径: {pdf_path}")

                    # 使用大模型总结论文 (使用run_in_executor将同步函数转为异步)
                    loop = asyncio.get_running_loop()
                    summary = await loop.run_in_executor(
                        None,
                        lambda: summarize_paper(pdf_content)
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

            # 辅助函数，将论文的内容总结写入到文件中
            def write_summary_to_file(path, title, content):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {title} - 论文总结\n\n")
                    f.write(strip_markdown_fences(content))

            # 首先顺序读取所有PDF文件的内容
            pdf_contents = {}
            for pdf_path in pdf_files:
                base_name = os.path.basename(pdf_path)
                print(f"\033[96m正在读取PDF: {base_name}\033[0m")
                logger.info(f"正在读取PDF: {base_name}, 路径: {pdf_path}")
                
                try:
                    # 读取PDF内容
                    middle_save_dir = os.path.join(PAPER_ANALYSIS_DIR, "middle")
                    pdf_content_result = read_pdf_content_local_single(pdf_path, output_dir=middle_save_dir)
                    
                    if pdf_content_result['status'] == 'success':
                        pdf_contents[pdf_path] = pdf_content_result['content']
                        print(f"\033[92mPDF '{base_name}' 读取成功\033[0m")
                        logger.info(f"PDF '{base_name}' 读取成功")
                    else:
                        print(f"\033[91mPDF '{base_name}' 读取失败: {pdf_content_result['content']}\033[0m")
                        logger.error(f"读取PDF内容失败: {pdf_content_result['content']}")
                except Exception as e:
                    print(f"\033[91mPDF '{base_name}' 读取出现异常: {str(e)}\033[0m")
                    logger.error(f"读取PDF异常: {str(e)}")

            # 使用信号量限制并发量
            semaphore = asyncio.Semaphore(config['max_concurrent_process'])  # 最多同时处理N篇论文分析

            # 只对成功读取内容的PDF文件进行分析
            analyzed_pdf_paths = list(pdf_contents.keys())
            print(f"\033[94m共有 {len(analyzed_pdf_paths)}/{len(pdf_files)} 个PDF文件读取成功，开始并发分析...\033[0m")
            logger.info(f"共有 {len(analyzed_pdf_paths)}/{len(pdf_files)} 个PDF文件读取成功，开始并发分析")

            async def process_with_semaphore(pdf_path):
                """使用信号量限制并发处理数量"""
                async with semaphore:
                    return await process_paper(pdf_path, pdf_contents[pdf_path])

            # 创建所有论文的处理任务
            tasks = [process_with_semaphore(pdf_path) for pdf_path in analyzed_pdf_paths]
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 过滤出正常结果和异常
            filtered_results = []
            for result in results:
                if isinstance(result, BaseException):
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
            with open(os.path.join(project_dir, "summary_report.json"), "w", encoding="utf-8") as f:
                json.dump(summary_report, f, ensure_ascii=False, indent=4)

            print(f"\033[94m论文分析完成: 总共 {summary_report['total']} 篇, 成功 {summary_report['success']} 篇, 失败 {summary_report['failed']} 篇\033[0m")
            logger.info(f"论文分析完成: 总共 {summary_report['total']} 篇, 成功 {summary_report['success']} 篇, 失败 {summary_report['failed']} 篇")

            # 创建综述报告的文件名
            report_filename = os.path.join(project_dir, config["paths"]["report_file"])

            # 收集成功总结的论文
            successful_papers = [r for r in filtered_results if r["status"] == "成功"]

            if successful_papers:
                summaries = []
                paper_titles = []
                for paper in successful_papers:
                    try:
                        with open(paper["output_path"], "r", encoding="utf-8") as f:
                            summaries.append(f.read())
                            paper_titles.append(paper["title"])
                    except Exception as e:
                        logger.error(f"读取论文总结失败: {paper['output_path']}, 错误: {str(e)}")

                # 生成综合分析报告
                comprehensive_report = generate_comprehensive_report(summaries, paper_titles)

                # 保存综述报告
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(strip_markdown_fences(comprehensive_report))

                print(f"\033[92m综述报告已成功创建: {report_filename}\033[0m")
                logger.info(f"综述报告已成功创建: {report_filename}")
            else:
                logger.warning("没有成功总结的论文，无法创建综述报告")
                print("\033[93m没有成功总结的论文，无法创建综述报告\033[0m")

            if step == 'all' and 'download_history' in locals():
                analyze_chat_history(download_history, "download")

            print(f"\033[94m所有任务已完成，结果已保存\033[0m")

            # 生成并且打开可视化网页
            file_list = []
            for md_file in os.listdir(PAPER_ANALYSIS_DIR):
                if md_file.endswith('.md'):
                    file_list.append({
                        'name': md_file,
                        'path': os.path.join(PAPER_ANALYSIS_DIR, md_file)
                    })
            with open('./file-list.json', 'w', encoding='utf-8') as f:
                json.dump(file_list, f, ensure_ascii=False, indent=4)

            # 在后台启动服务器
            threading.Thread(
                target=start_server,
                args=(PAPER_ANALYSIS_DIR, config["front_end"]["port"]),
                daemon=True
            ).start()

            # 打开浏览器
            webbrowser.open(f'http://localhost:{config["front_end"]["port"]}/index.html')

            # 保持主线程运行
            try:
                while True:
                    pass
            except KeyboardInterrupt:
                print("\n关闭服务器")
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
        parser.add_argument('--project-dir', type=str, default='D:\\owl\\papers',
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
