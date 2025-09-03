#    Modified from https://github.com/haotian-liu/LLaVA
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from PIL import Image
from io import BytesIO
import requests
import os
import base64


def load_image_from_base64(image):
    """
    从 Base64 编码的字符串中加载图像。

    Args:
        image (str): Base64 编码的图像字符串。

    Returns:
        PIL.Image.Image: 解码后的图像对象。
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_file):
    """
    从文件路径或 URL 加载图像。

    Args:
        image_file (str): 图像文件路径或 URL。

    Returns:
        PIL.Image.Image: 加载的 RGB 图像。
    """
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    """
    批量加载图像。

    Args:
        image_files (list): 图像文件路径或 URL 的列表。

    Returns:
        list: 包含所有加载图像的列表。
    """
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def vis_images(image_files):
    """
    可视化图像。

    Args:
        image_files (list): 图像文件路径的列表。

    功能:
        - 如果只有一张图像，直接显示。
        - 如果有多张图像，将它们拼接成一张图像后显示。
    """
    if len(image_files) == 1:
        image = image_files[0]
        os.system(
            f"termvisage --query-timeout 1 -H left --height 40 --oversize {image}"
        )  # --height 50：设置图片高度为 500 行。

    else:
        # 拼接多张图像
        system_inst = "convert "
        inst_template1 = " \\( {image} -background none -resize x{height} \\) "
        inst_template2 = (
            " \\( {image} -background none -resize x{height} -splice 50x0 \\) "
        )
        count = 0
        for image in image_files:
            with Image.open(image) as img:
                width, height = img.size  # 查看尺寸
                print(f"{image} width and height is {width}, {height}")

            count += 1
            if count == 1:
                system_inst += inst_template1.format(image=image, height=height)
            else:
                system_inst += inst_template2.format(image=image, height=height)
        system_inst += " +append .vis.jpg"
        os.system(system_inst)

        os.system(f"termvisage --query-timeout 1 .vis.jpg -H left")


def expand2square(pil_img, background_color):
    """
    将图像扩展为正方形。

    Args:
        pil_img (PIL.Image.Image): 输入的图像。
        background_color (tuple): 背景颜色，格式为 RGB。

    Returns:
        PIL.Image.Image: 扩展后的正方形图像。
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    """
    处理图像以适配模型输入。

    Args:
        images (list): 图像列表。
        image_processor: 图像处理器对象。
        model_cfg: 模型配置对象。

    Returns:
        torch.Tensor 或 list: 处理后的图像张量或列表。
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            if "intern" in image_processor.__class__.__name__.lower():
                # 特殊情况
                new_images.append(image.unsqueeze(0))
            else:
                new_images.append(image)
    else:
        ret = image_processor(images, return_tensors="pt")["pixel_values"]
        if "intern" in image_processor.__class__.__name__.lower():
            # 特殊情况
            ret = [x.unsqueeze(0) for x in ret]
        return ret
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)

    return new_images
