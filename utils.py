import os
import logging
import requests
import time
import glob
import zipfile
import io
import json
import re
from pathlib import Path
from camel.logger import set_log_level, get_logger
from config_loader import ConfigLoader

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
def read_pdf_content(pdf_path: str, is_directory: bool = False, output_dir: str = "./output") -> dict:
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


def read_pdf_content_local(pdf_path: str, is_directory: bool = False, output_dir: str = "./output") -> dict:
    """
    使用本地安装的 MineU (magic_pdf) 库处理单个或多个 PDF 文件，提取 Markdown 内容。

    Args:
        pdf_path: 指向单个 PDF 文件或包含 PDF 文件的目录的路径。
        is_directory: 布尔值，指示 `pdf_path` 是否是目录。默认为 False。
        output_dir: 用于存储中间文件（如图片、布局调试 PDF）和最终 Markdown 文件的目录。
                     默认为 "./output"。

    Returns:
        dict:
            - 如果 `is_directory` 为 True: 返回一个字典，键是原始 PDF 文件名，
              值是包含处理结果的字典 `{"content": str, "status": str}`。
              `content` 是提取的 Markdown 内容（成功时）或错误信息（失败时），
              `status` 是 "success" 或 "fail"。
            - 如果 `is_directory` 为 False: 返回单个包含处理结果的字典
              `{"content": str, "status": str}`。
            - 如果在初始阶段发生错误（例如库导入失败、文件/目录未找到）或捕获到主要异常，
              则返回包含错误信息的字典 `{"content": str, "status": "fail"}`。
              (若为目录处理且发生全局异常但有部分成功结果，会尝试返回部分结果)。

    Raises:
        无显式抛出，但依赖的库可能会抛出异常。函数内部会尝试捕获常见异常。

    Note:
        - 需要本地正确安装 `magic_pdf` 相关的库。
        - 处理结果（包括 Markdown 和中间文件）会保存在 `output_dir` 中。
    """
    try:
        import os
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod

        os.makedirs(output_dir, exist_ok=True)
        local_image_dir = os.path.join(output_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        image_dir = os.path.basename(local_image_dir)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(output_dir)

        pdf_files = []
        file_names = []
        result_dict = {} # 初始化 result_dict

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

        reader = FileBasedDataReader("")

        for i, (pdf_file, file_name) in enumerate(zip(pdf_files, file_names)):
            try:
                logger.info(f"开始处理文件 {file_name}")
                name_without_suffix = os.path.splitext(file_name)[0]

                # 读取PDF文件
                pdf_bytes = reader.read(pdf_file)

                # 创建数据集实例
                ds = PymuDocDataset(pdf_bytes)

                # 推理
                if ds.classify() == SupportedPdfParseMethod.OCR:
                    logger.info(f"文件 {file_name} 使用OCR模式处理")
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    logger.info(f"文件 {file_name} 使用文本模式处理")
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                # 保存模型结果 (可选)
                # infer_result.draw_model(os.path.join(output_dir, f"{name_without_suffix}_model.pdf"))

                # 获取模型推理结果 (可选)
                # model_inference_result = infer_result.get_infer_res()

                # 保存布局结果 (可选)
                # pipe_result.draw_layout(os.path.join(output_dir, f"{name_without_suffix}_layout.pdf"))

                # 保存spans结果 (可选)
                # pipe_result.draw_span(os.path.join(output_dir, f"{name_without_suffix}_spans.pdf"))

                # 获取markdown内容
                md_content_from_pipe = pipe_result.get_markdown(image_dir)

                # 保存markdown
                md_file_path = f"{name_without_suffix}.md"
                pipe_result.dump_md(md_writer, md_file_path, image_dir)

                # 生成和保存内容列表 (可选)
                # content_list_content = pipe_result.get_content_list(image_dir)
                # pipe_result.dump_content_list(md_writer, f"{name_without_suffix}_content_list.json", image_dir)

                # 生成和保存中间JSON (可选)
                # middle_json_content = pipe_result.get_middle_json()
                # pipe_result.dump_middle_json(md_writer, f'{name_without_suffix}_middle.json')

                # 再次读取生成的markdown文件以确保内容一致性 (如果 dump_md 可靠，也可直接用 md_content_from_pipe)
                md_file_full_path = os.path.join(output_dir, md_file_path)
                if os.path.exists(md_file_full_path):
                    with open(md_file_full_path, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    result_dict[file_name] = {
                        "content": md_content,
                        "status": "success"
                    }
                    logger.info(f"文件 {file_name} 处理成功，共{len(md_content)}字符")
                else:
                    # 如果dump_md失败或未生成文件
                    error_message = f"处理文件 {file_name} 成功，但未能读取生成的Markdown文件: {md_file_full_path}"
                    logger.error(error_message)
                    result_dict[file_name] = {
                        "content": md_content_from_pipe if md_content_from_pipe else error_message, # 尝试用内存中的md内容
                        "status": "fail" # 标记为失败因为文件未找到
                    }

            except Exception as e:
                error_message = str(e)
                logger.error(f"处理文件 {file_name} 时出错: {error_message}")
                result_dict[file_name] = {
                    "content": f"处理失败: {error_message}",
                    "status": "fail"
                }

        # 根据 is_directory 返回最终结果
        if is_directory:
            return result_dict
        else:
            # 对于单个文件，返回其对应的结果字典
            # 如果 result_dict 为空 (例如初始检查就失败了，虽然上面已经return了，但作为保障)
            # 或者处理失败了
            return result_dict.get(file_names[0], {"content": "处理失败或未找到结果", "status": "fail"})

    except ImportError as e:
        error_message = f"未能导入必要的MineU库: {str(e)}. 请确保已正确安装 magic_pdf 相关依赖。"
        logger.error(error_message)
        return {"content": error_message, "status": "fail"}
    except Exception as e:
        error_message = f"PDF本地批量解析过程中发生未预料的错误: {str(e)}"
        logger.exception(error_message) # 使用 exception 记录堆栈信息
        # 检查是否是目录处理且已有部分结果
        if is_directory and result_dict:
            logger.warning(f"发生未预料错误，但已有部分文件处理完毕({len(result_dict)}/{len(file_names)})，将返回现有结果。")
            # 补充未处理文件的错误状态
            for file_name in file_names:
                if file_name not in result_dict:
                    result_dict[file_name] = {"content": f"因全局错误未处理: {error_message}", "status": "fail"}
            return result_dict
        # 如果是单个文件或目录处理但无任何结果，返回全局错误
        return {"content": error_message, "status": "fail"}


def test_read_pdf_content_local():
    result = read_pdf_content_local("D:/test/mineru/AutoSurvey.pdf", False, "./test/output")
    print(result)


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