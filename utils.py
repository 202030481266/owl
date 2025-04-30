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
# todo 接入本地的 MineU ，速度会得到大幅提升
def read_pdf_content(pdf_path: str, is_directory: bool = False, output_dir: str = "./output") -> dict:
    api_key = os.getenv("MINERU_API_KEY")
    if not api_key:
        logger.error("未找到MINERU_API_KEY环境变量")
        return {"error": "未找到MINERU_API_KEY环境变量"}

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = []
    file_names = []

    if is_directory:
        if not os.path.isdir(pdf_path):
            logger.error(f"目录不存在: {pdf_path}")
            return {"error": f"目录不存在: {pdf_path}"}

        pdf_pattern = os.path.join(pdf_path, "*.pdf")
        for pdf_file in glob.glob(pdf_pattern):
            pdf_files.append(pdf_file)
            file_names.append(os.path.basename(pdf_file))

        if not pdf_files:
            logger.error(f"目录中未找到PDF文件: {pdf_path}")
            return {"error": f"目录中未找到PDF文件: {pdf_path}"}
    else:
        if not os.path.exists(pdf_path):
            logger.error(f"文件不存在: {pdf_path}")
            return {"error": f"文件不存在: {pdf_path}"}

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
            logger.error(f"请求上传URL失败: {response.status_code}, {response.text}")
            return {"error": f"请求上传URL失败: {response.status_code}"}

        result = response.json()
        if result.get("code") != 0:
            error_msg = result.get("msg", "未知错误")
            logger.error(f"申请上传URL失败: {error_msg}")
            return {"error": f"申请上传URL失败: {error_msg}"}

        batch_id = result["data"]["batch_id"]
        if "file_urls" not in result["data"]:
            logger.error(f"响应中缺少file_urls字段: {result}")
            return {"error": "响应格式不符合预期，缺少file_urls字段"}

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
                logger.error(f"获取结果失败: {res.status_code}, {res.text}")
                if len(result_dict) > 0:
                    return result_dict
                return {"error": f"获取结果失败: {res.status_code}"}

            result_data = res.json()

            if result_data.get("code") != 0:
                logger.error(f"获取结果失败: {result_data.get('msg', '未知错误')}")
                continue

            extract_result = result_data.get("data", {}).get("extract_result", [])
            if not extract_result:
                logger.warning("响应中没有extract_result字段或为空")
                time.sleep(3)
                continue

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

                if state == "done":
                    zip_url = task.get("full_zip_url", "")
                    if zip_url:
                        try:
                            zip_response = requests.get(zip_url)
                            if zip_response.status_code == 200:
                                base_name = os.path.splitext(file_name)[0]
                                output_file = os.path.join(output_dir, f"{base_name}.zip")

                                with open(output_file, 'wb') as f:
                                    f.write(zip_response.content)

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
                                    logger.error(f"提取full.md内容时发生错误: {str(ex)}")
                                    result_dict[file_name] = {
                                        "content": f"提取full.md内容时发生错误: {str(ex)}",
                                        "status": "fail"
                                    }
                                    processed_files.add(file_name)
                            else:
                                logger.error(f"下载解析结果ZIP失败: {zip_response.status_code}")
                                result_dict[file_name] = {
                                    "content": f"下载解析结果失败: {zip_response.status_code}",
                                    "status": "fail"
                                }
                                processed_files.add(file_name)
                        except Exception as e:
                            logger.error(f"下载ZIP文件时出错: {str(e)}")
                            result_dict[file_name] = {
                                "content": f"下载ZIP文件时出错: {str(e)}",
                                "status": "fail"
                            }
                            processed_files.add(file_name)
                    else:
                        result_dict[file_name] = {
                            "content": "解析完成但未返回下载链接",
                            "status": "fail"
                        }
                        processed_files.add(file_name)
                        logger.warning(f"文件 {file_name} 解析完成但未返回下载链接")
                elif state == "failed":
                    error_message = task.get("err_msg", "未知错误")
                    result_dict[file_name] = {
                        "content": f"处理失败: {error_message}",
                        "status": "fail"
                    }
                    processed_files.add(file_name)
                    logger.error(f"文件 {file_name} 处理失败: {error_message}")
                else:
                    all_done = False
                    progress_info = task.get("extract_progress", {})
                    if progress_info:
                        extracted = progress_info.get("extracted_pages", 0)
                        total = progress_info.get("total_pages", 0)
                        if total > 0:
                            logger.info(f"文件 {file_name} 处理进度: {extracted}/{total} 页")

            if len(processed_files) >= len(pdf_files) or all_done:
                break

            time.sleep(3)

        for file_name in file_names:
            if file_name not in processed_files:
                result_dict[file_name] = {
                    "content": "处理超时或未返回结果",
                    "status": "fail"
                }
                logger.warning(f"文件 {file_name} 处理超时或未返回结果")

        return result_dict if is_directory else next(iter(result_dict.values()))

    except Exception as e:
        logger.error(f"PDF批量解析过程中发生错误: {str(e)}")
        return {"error": f"PDF批量解析过程中发生错误: {str(e)}"}


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