import base64
import os
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI

from .utils import Config

gen_config = Config().get_config()

# 禁用当前进程的系统代理
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)


client = OpenAI(base_url="http://127.0.0.1:8045/v1", api_key="sk-a1c626e503774854bd354e27ae92b2dc")


def simple_llm_generate():
    try:
        response = client.chat.completions.create(model="gemini-3-pro-high", messages=[{"role": "user", "content": "你好，介绍下你自己"}])
        logger.info(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        logger.error("请确保本地 API 服务运行在 http://127.0.0.1:8045")
        raise


def simple_img_generate():
    try:
        response = client.chat.completions.create(
            model="gemini-3-pro-image",
            # 方式 1: 使用 size 参数 (推荐)
            # 支持: "1024x1024" (1:1), "1280x720" (16:9), "720x1280" (9:16), "1216x896" (4:3)
            extra_body={"size": "1024x1024"},
            # 方式 2: 使用模型后缀
            # 例如: gemini-3-pro-image-16-9, gemini-3-pro-image-4-3
            # model="gemini-3-pro-image-16-9",
            messages=[{"role": "user", "content": "Draw a futuristic city"}],
        )

        # 获取图片内容（假设返回的是 base64 编码或 URL）
        image_content = response.choices[0].message.content
        logger.info(f"图像生成成功: {image_content[:100]}...")

        # 创建 outputs 目录
        output_dir = Path(gen_config["outputs"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 处理 Markdown 格式的图片：![image](data:image/jpeg;base64,...)
        if image_content.startswith("![image](") and image_content.endswith(")"):
            image_content = image_content[len("![image](") : -1]

        # 如果是 base64 编码的图片，解码并保存
        if image_content.startswith("data:image"):
            # 处理 data:image/png;base64,... 或 data:image/jpeg;base64,... 格式
            # 提取文件扩展名
            if "image/jpeg" in image_content or "image/jpg" in image_content:
                ext = "jpg"
            elif "image/png" in image_content:
                ext = "png"
            else:
                ext = "png"

            output_path = output_dir / f"generated_image_{timestamp}.{ext}"
            base64_data = image_content.split(",", 1)[1]
            image_data = base64.b64decode(base64_data)
            output_path.write_bytes(image_data)
            logger.info(f"图片已保存到: {output_path}")
        elif image_content.startswith("http"):
            # 如果是 URL，需要下载
            import requests

            output_path = output_dir / f"generated_image_{timestamp}.png"
            img_response = requests.get(image_content)
            output_path.write_bytes(img_response.content)
            logger.info(f"图片已下载并保存到: {output_path}")
        else:
            # 直接当作二进制数据保存
            output_path = output_dir / f"generated_image_{timestamp}.txt"
            logger.warning(f"未知的图片格式，尝试直接保存: {image_content[:50]}")
            output_path.write_text(image_content)

        return str(output_path)
    except Exception as e:
        logger.error(f"图像生成调用失败: {e}")
        logger.error("请确保本地 API 服务运行在 http://127.0.0.1:8045")
        raise
