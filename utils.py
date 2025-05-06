import os
import logging
import requests
import time
import glob
import zipfile
import io
import json
import re
import traceback
from pathlib import Path
from camel.logger import set_log_level, get_logger
from config_loader import ConfigLoader

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# 加载配置
config = ConfigLoader.load_config()

# 设置日志
set_log_level(level=config["logging"]["level"])
logger = get_logger(__name__)
file_handler = logging.FileHandler(config["logging"]["utils_file"], encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(config["logging"]["format"])
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
utils_logger = logging.getLogger()
utils_logger.addHandler(file_handler)


# 读取pdf文件的内容
def read_pdf_content_api_single(pdf_path: str, is_directory: bool = False, output_dir: str = "./output") -> dict:
    """
    使用 MinerU API 处理单个 PDF 文件或目录中的多个 PDF 文件，提取 Markdown 内容。

    该函数会首先向 MinerU API 请求上传 URL，然后上传文件，接着轮询 API 获取处理结果。
    处理成功后，会下载包含 Markdown 文件的 ZIP 压缩包，并从中提取 `full.md` 的内容。

    Args:
        pdf_path: 指向单个 PDF 文件或包含 PDF 文件的目录的路径。
        is_directory: 布尔值，指示 `pdf_path` 是否是目录。默认为 False。
        output_dir: 用于存储下载的 ZIP 结果文件的目录。默认为 "./output"。

    Returns:
        dict:
            - 如果 `is_directory` 为 True: 返回一个字典，键是原始 PDF 文件名，
              值是包含处理结果的字典 `{"content": str, "status": str}`。
              `content` 是提取的 Markdown 内容（成功时）或错误信息（失败时），
              `status` 是 "success" 或 "fail"。
            - 如果 `is_directory` 为 False: 返回单个包含处理结果的字典
              `{"content": str, "status": str}`。
            - 如果在初始阶段发生错误（例如 API 密钥缺失、文件/目录未找到、
              API 请求失败）或捕获到主要异常，则返回包含错误信息的字典
              `{"content": str, "status": "fail"}`。

    Raises:
        无显式抛出，但依赖的库（如 requests）可能会抛出异常。函数内部会尝试捕获常见异常。

    Note:
        - 此函数需要设置 `MINERU_API_KEY` 环境变量。
        - API 端点和其他配置从 `config_loader` 加载的配置中读取。
        - 处理时间和成功率取决于 MinerU API 的性能和状态。
    """
    api_key = os.getenv("MINERU_API_KEY")
    if not api_key:
        error_message = "未找到MINERU_API_KEY环境变量"
        logger.error(error_message)
        return {"content": error_message, "status": "fail"}

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = []
    file_names = []

    if is_directory:
        if not os.path.isdir(pdf_path):
            error_message = f"目录不存在: {pdf_path}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}

        pdf_pattern = os.path.join(pdf_path, "*.pdf")
        for pdf_file in glob.glob(pdf_pattern):
            pdf_files.append(pdf_file)
            file_names.append(os.path.basename(pdf_file))

        if not pdf_files:
            error_message = f"目录中未找到PDF文件: {pdf_path}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}
    else:
        if not os.path.exists(pdf_path):
            error_message = f"文件不存在: {pdf_path}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}

        pdf_files.append(pdf_path)
        file_names.append(os.path.basename(pdf_path))

    upload_url = config["api"]["mineru"]["upload_url"]
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {api_key}"
    }

    files_data = []
    for i, file_name in enumerate(file_names):
        files_data.append({
            "name": file_name,
            "is_ocr": False,
            "data_id": file_name
        })

    data = {
        "enable_formula": config["api"]["mineru"]["enable_formula"],
        "language": config["api"]["mineru"]["language"],
        "layout_model": config["api"]["mineru"]["layout_model"],
        "enable_table": config["api"]["mineru"]["enable_table"],
        "files": files_data
    }

    try:
        response = requests.post(upload_url, headers=headers, json=data)
        if response.status_code != 200:
            error_message = f"请求上传URL失败: {response.status_code}, {response.text}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}

        result = response.json()
        if result.get("code") != 0:
            error_msg = result.get("msg", "未知错误")
            error_message = f"申请上传URL失败: {error_msg}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}

        batch_id = result["data"]["batch_id"]
        if "file_urls" not in result["data"]:
            error_message = f"响应中缺少file_urls字段: {result}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}

        upload_urls = result["data"]["file_urls"]

        for i, (pdf_file, upload_url) in enumerate(zip(pdf_files, upload_urls)):
            with open(pdf_file, 'rb') as f:
                res_upload = requests.put(upload_url, data=f)
                if res_upload.status_code != 200:
                    logger.error(f"文件 {file_names[i]} 上传失败: {res_upload.status_code}")
                else:
                    logger.info(f"文件 {file_names[i]} 上传成功")

        logger.info(f"所有文件上传完成，批处理ID: {batch_id}")

        max_attempts = 200 * len(pdf_files)
        attempt = 0
        result_dict = {}
        processed_files = set()

        while attempt < max_attempts and len(processed_files) < len(pdf_files):
            attempt += 1

            result_url = config["api"]["mineru"]["result_url_template"].format(batch_id)
            res = requests.get(result_url, headers=headers)

            if res.status_code != 200:
                error_message = f"获取结果失败: {res.status_code}, {res.text}"
                logger.error(error_message)
                if is_directory and len(result_dict) > 0:
                    # 如果是目录处理且已有部分结果，返回部分结果，不覆盖全局错误
                    logger.warning("获取结果时出错，但已有部分文件处理完毕，将返回现有结果。")
                    return result_dict
                return {"content": error_message, "status": "fail"}

            result_data = res.json()

            if result_data.get("code") != 0:
                error_message = f"获取结果API返回错误: {result_data.get('msg', '未知错误')}"
                logger.error(error_message)
                # 不在此处返回，继续尝试或等待超时
                time.sleep(3)
                continue

            extract_result = result_data.get("data", {}).get("extract_result", [])
            if not extract_result:
                logger.warning("响应中没有extract_result字段或为空，可能是处理中，将继续轮询")
                time.sleep(3)
                continue

            all_done_in_batch = True
            for task in extract_result:
                if not isinstance(task, dict):
                    logger.error(f"收到意外的响应格式，task不是字典: {task}")
                    continue

                file_name = task.get("file_name", "")
                state = task.get("state", "")

                if not file_name or file_name not in file_names:
                    # 可能返回了不属于本次请求的文件信息，跳过
                    continue

                if file_name in processed_files:
                    # 已处理的文件，跳过
                    continue

                if state == "done":
                    zip_url = task.get("full_zip_url", "")
                    if zip_url:
                        try:
                            zip_response = requests.get(zip_url)
                            if zip_response.status_code == 200:
                                base_name = os.path.splitext(file_name)[0]
                                output_file = os.path.join(output_dir, f"{base_name}.zip")

                                # 保存zip文件 (可选，如果需要保留)
                                # with open(output_file, 'wb') as f:
                                #     f.write(zip_response.content)

                                try:
                                    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zf:
                                        md_files = [f for f in zf.namelist() if f.endswith('full.md')]
                                        if md_files:
                                            with zf.open(md_files[0]) as md_file:
                                                md_content = md_file.read().decode('utf-8')
                                                result_dict[file_name] = {
                                                    "content": md_content,
                                                    "status": "success"
                                                }
                                                logger.info(f"已从文件 {file_name} 中提取full.md内容，共{len(md_content)}字符")
                                        else:
                                            logger.warning(f"ZIP文件中未找到full.md: {output_file}")
                                            result_dict[file_name] = {
                                                "content": f"处理成功但未找到full.md文件: {output_file}",
                                                "status": "fail"
                                            }
                                    processed_files.add(file_name)
                                except Exception as ex:
                                    error_message = f"提取full.md内容时发生错误: {str(ex)}"
                                    logger.error(error_message)
                                    result_dict[file_name] = {
                                        "content": error_message,
                                        "status": "fail"
                                    }
                                    processed_files.add(file_name)
                            else:
                                error_message = f"下载解析结果ZIP失败: {zip_response.status_code}"
                                logger.error(error_message)
                                result_dict[file_name] = {
                                    "content": error_message,
                                    "status": "fail"
                                }
                                processed_files.add(file_name)
                        except Exception as e:
                            error_message = f"下载ZIP文件时出错: {str(e)}"
                            logger.error(error_message)
                            result_dict[file_name] = {
                                "content": error_message,
                                "status": "fail"
                            }
                            processed_files.add(file_name)
                    else:
                        error_message = "解析完成但未返回下载链接"
                        result_dict[file_name] = {
                            "content": error_message,
                            "status": "fail"
                        }
                        processed_files.add(file_name)
                        logger.warning(f"文件 {file_name} {error_message}")
                elif state == "failed":
                    error_message = task.get("err_msg", "未知错误")
                    result_dict[file_name] = {
                        "content": f"处理失败: {error_message}",
                        "status": "fail"
                    }
                    processed_files.add(file_name)
                    logger.error(f"文件 {file_name} 处理失败: {error_message}")
                else: # running, pending, etc.
                    all_done_in_batch = False
                    progress_info = task.get("extract_progress", {})
                    if progress_info:
                        extracted = progress_info.get("extracted_pages", 0)
                        total = progress_info.get("total_pages", 0)
                        if total > 0:
                            logger.info(f"文件 {file_name} 处理进度: {extracted}/{total} 页")

            if len(processed_files) >= len(pdf_files) or all_done_in_batch:
                # 如果所有文件都已记录状态(成功或失败)，或者API报告本次查询的所有任务都已完成，则退出轮询
                break

            time.sleep(3) # 轮询间隔

        # 轮询结束后，检查是否有文件未被处理 (超时)
        for file_name in file_names:
            if file_name not in processed_files:
                error_message = "处理超时或未返回最终结果"
                result_dict[file_name] = {
                    "content": error_message,
                    "status": "fail"
                }
                logger.warning(f"文件 {file_name} {error_message}")

        # 根据 is_directory 返回最终结果
        if is_directory:
            return result_dict
        else:
            # 对于单个文件，返回其对应的结果字典
            # 如果 result_dict 为空 (例如上传后立刻出错)，则返回一个通用的失败状态
            return result_dict.get(file_names[0], {"content": "处理失败或未返回结果", "status": "fail"})

    except Exception as e:
        error_message = f"PDF批量解析过程中发生未预料的错误: {str(e)}"
        logger.exception(error_message) # 使用 exception 记录堆栈信息
        # 如果是目录处理且已有部分结果，可能需要考虑是否返回部分结果
        if is_directory and result_dict:
            logger.warning(f"发生未预料错误，但已有部分文件处理完毕({len(result_dict)}/{len(pdf_files)})，将返回现有结果。")
            # 补充未处理文件的错误状态
            for file_name in file_names:
                if file_name not in result_dict:
                    result_dict[file_name] = {"content": f"因全局错误未处理: {error_message}", "status": "fail"}
            return result_dict
        # 如果是单个文件或目录处理但无任何结果，返回全局错误
        return {"content": error_message, "status": "fail"}


def read_pdf_content_local_single(pdf_path: str, output_dir: str) -> dict:
    """
    使用本地安装的 MineU (magic_pdf) 库处理单个 PDF 文件，提取 Markdown 内容。
    Args:
        pdf_path: 指向单个 PDF 文件的路径。
        output_dir: 用于存储中间文件（如图片、布局调试 PDF）和最终 Markdown 文件的目录。
                     默认为 "./output"。
    Returns:
        dict:
            - 返回一个字典，键是原始 PDF 文件名，
              值是包含处理结果的字典 `{"content": str, "status": str}`。
    """
    # 如果pdf_path不存在，则返回错误
    if not os.path.exists(pdf_path):
        error_message = f"文件不存在: {pdf_path}"
        logger.error(error_message)
        return {"content": error_message, "status": "fail"}

    # 创建文件夹，保存输出的结果和中间件
    os.makedirs(output_dir, exist_ok=True)
    local_image_dir = os.path.join(output_dir, "images")
    os.makedirs(local_image_dir, exist_ok=True) # 创建图片保存的文件夹
    image_dir = os.path.basename(local_image_dir)

    # 创建文件写入器
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(output_dir)

    # 读取PDF文件
    pdf_reader = FileBasedDataReader("")
    filename = os.path.basename(pdf_path) # 获取文件名
    name_without_suffix = os.path.splitext(filename)[0]

    # 读取PDF文件
    pdf_bytes = pdf_reader.read(pdf_path)

    # 创建数据集实例
    ds = PymuDocDataset(pdf_bytes)

    try:
        # 推理
        if ds.classify() == SupportedPdfParseMethod.OCR:
            logger.info(f"文件 {filename} 使用OCR模式处理")
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            logger.info(f"文件 {filename} 使用文本模式处理")
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
    except Exception as e:
        error_details = traceback.format_exc()  # 获取完整的堆栈跟踪
        error_message = f"处理文件 {filename} 时在推理阶段发生错误: {str(e)}\n堆栈跟踪: \n{error_details}"
        logger.error(error_message)
        return {"content": error_message, "status": "fail"}
    else:
        try:
            # 保存模型结果 (可选)
            infer_result.draw_model(os.path.join(output_dir, f"{name_without_suffix}_model.pdf"))

            # 保存布局结果 (可选)
            pipe_result.draw_layout(os.path.join(output_dir, f"{name_without_suffix}_layout.pdf"))

            # 保存spans结果 (可选)
            pipe_result.draw_span(os.path.join(output_dir, f"{name_without_suffix}_spans.pdf"))

            # 获取markdown内容
            md_content_from_pipe = pipe_result.get_markdown(image_dir)

            # 保存markdown
            md_file_path = f"{name_without_suffix}.md"
            pipe_result.dump_md(md_writer, md_file_path, image_dir)

            # 生成和保存内容列表 (可选)
            pipe_result.dump_content_list(md_writer, f"{name_without_suffix}_content_list.json", image_dir)

            # 生成和保存中间JSON (可选)
            pipe_result.dump_middle_json(md_writer, f'{name_without_suffix}_middle.json')
        except Exception as e:
            error_details = traceback.format_exc()  # 获取完整的堆栈跟踪
            error_message = f"处理文件 {filename} 时在保存结果阶段发生错误: {str(e)}\n堆栈跟踪: \n{error_details}"
            logger.error(error_message)
            return {"content": error_message, "status": "fail"}
        else:
            return {"content": md_content_from_pipe, "status": "success"}


def read_pdf_content_local_batch(pdf_paths: list[str], output_dir: str = "./output") -> list[dict]:
    """
    使用本地安装的 MineU (magic_pdf) 库处理单个或多个 PDF 文件，提取 Markdown 内容。

    Args:
        pdf_paths: 包含多个 PDF 文件路径的列表。
        output_dir: 用于存储中间文件（如图片、布局调试 PDF）和最终 Markdown 文件的目录。
                     默认为 "./output"。

    Returns:
        dict:
            - 返回一个字典，键是原始 PDF 文件名，
              值是包含处理结果的字典 `{"content": str, "status": str}`。
              `content` 是提取的 Markdown 内容（成功时）或错误信息（失败时），
              `status` 是 "success" 或 "fail"。
            - 如果在初始阶段发生错误（例如库导入失败、文件/目录未找到）或捕获到主要异常，
              则返回包含错误信息的字典 `{"content": str, "status": "fail"}`。
              (若为目录处理且发生全局异常但有部分成功结果，会尝试返回部分结果)。

    Raises:
        无显式抛出，但依赖的库可能会抛出异常。函数内部会尝试捕获常见异常。

    Note:
        - 需要本地正确安装 `magic_pdf` 相关的库。
        - 处理结果（包括 Markdown 和中间文件）会保存在 `output_dir` 中。
    """
    results = []
    for pdf_path in pdf_paths:
        try:
            pdf_file_name = os.path.basename(pdf_path)
            name_without_suffix = os.path.splitext(pdf_file_name)[0]
            output_dir_path = os.path.join(output_dir, name_without_suffix)
            os.makedirs(output_dir_path, exist_ok=True)
            result = read_pdf_content_local_single(pdf_path, output_dir_path)
            results.append(result)
        except Exception as e:
            error_message = f"处理文件 {pdf_file_name} 时发生错误: {str(e)}"
            logger.error(error_message)
            results.append({"content": error_message, "status": "fail"})
    return results


def test_read_pdf_content_local():
    # 测试单个文件处理
    single_result = read_pdf_content_local_single("D:/test/mineru/AutoSurvey.pdf", "D:/owl/test/output/AutoSurvey")
    
    # 测试批量文件处理
    batch_files = ["D:/test/mineru/AutoSurvey.pdf", "D:/test/mineru/paper2code.pdf", "D:/test/mineru/LLMxMapReduceV2.pdf"]
    batch_results = read_pdf_content_local_batch(batch_files, "D:/owl/test/output")
    
    # 检查批量处理结果
    all_batch_success = all(result["status"] == "success" for result in batch_results)
    
    # 在最后统一输出结果
    if single_result["status"] == "success":
        print("\033[38;5;46mread_pdf_content_local_single 测试通过！\033[0m")  # 更鲜艳的绿色
    else:
        print(f"\033[38;5;196mread_pdf_content_local_single 测试失败: {single_result['content']}\033[0m")  # 更鲜艳的红色
    
    if all_batch_success:
        print("\033[38;5;46mread_pdf_content_local_batch 测试通过！\033[0m")  # 更鲜艳的绿色
    else:
        failed_files = [batch_files[i] for i, result in enumerate(batch_results) if result["status"] == "fail"]
        print(f"\033[38;5;196mread_pdf_content_local_batch 测试失败，以下文件处理失败: {', '.join(failed_files)}\033[0m")  # 更鲜艳的红色

    
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


def strip_markdown_fences(text: str) -> str:
    """
    将所有 ```markdown ... ``` 包裹的区块去掉标记，只保留内部内容。
    支持多处出现、跨行匹配。
    同时处理不完整的标记，例如只有开始没有结束的情况。
    """
    # 首先处理完整的markdown标记对
    pattern = re.compile(r'```markdown\s*\n([\s\S]*?)\n```', re.MULTILINE)
    result = pattern.sub(r'\1', text)
    
    # 处理只有开始标记的情况
    incomplete_pattern = re.compile(r'```markdown\s*\n([\s\S]*?)$', re.MULTILINE)
    result = incomplete_pattern.sub(r'\1', result)
    
    return result


test_read_pdf_content_local()