import base64
import hashlib
from loguru import logger
from openai import OpenAI
import os
import requests

class OpenaiModelsCall:
    temp_audio_dir = "./audios" # 临时音频目录
    temp_image_dir = "./images"
    def __init__(self):
        self.client = OpenAI()
        self.max_token = 200
    # openai chat接口
    def chat_with_openai(self, prompt, history=[]):
        messages = [{"role": "system", "content": "你是一个聪明的助手"}]
        for h in history:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["assistant"]})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=self.max_token,
            stream=False,
        )
        
        return response.choices[0].message.content
    
    # 图像理解 
    def image_understanding(self, prompt: str, img_path: str) -> str | None:
        """
        调用 GPT-4o 视觉能力解析图片。
        - 小图（< 2 MB）可直接 base64；
        - 大图则自动上传到临时图床（此处用自建 upload_image() 占位）。
        """
        import uuid, requests, mimetypes
        from openai import BadRequestError, OpenAIError

        # 1) 读取文件
        with open(img_path, "rb") as f:
            data = f.read()


        mime = mimetypes.guess_type(img_path)[0] or "image/png"
        img_source = f"data:{mime};base64,{base64.b64encode(data).decode()}"

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",               # ← 新模型
                max_tokens=300,
                timeout=45,                   # openai-python ≥1.3 支持
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_source,
                                    "detail": "auto"     # low / high / auto
                                },
                            },
                        ],
                    }
                ],
            )
            return resp.choices[0].message.content

        except BadRequestError as e:
            logger.error(f"Bad request: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        return None

    # 语音转文字 todo:功能测试
    def speech_to_text(self,
                       audio_path: str,
                       model: str = "whisper-1",
                       language: str | None = None) -> str:
        """
        :param audio_path: 本地音频文件路径
        :param model: Whisper 模型，默认 'whisper-1'
        :param language: 可选，手动指定语种（如 'zh'、'en'）；留空则由模型自动检测
        :return: 识别后的纯文本
        """
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language
            )
        return response.text

    # 文本生成图像（使用 OpenAI DALL·E）  todo: 修改功能bug
    def generate_image(self, prompt: str,
                       model: str = "gpt-4o",
                       size: str = "1024x1024") -> str | None:
        """
        调用 DALL·E 生成图片并保存到本地

        :param prompt: 文字提示
        :param model: 生成模型，可选 "dall-e-3" 或 "dall-e-2"
        :param size: 生成尺寸，支持 256x256 / 512x512 / 1024x1024
        :return: 成功返回图片本地路径，失败返回 None
        """
        import uuid
        output_path = f"{OpenaiModelsCall.temp_image_dir}/{uuid.uuid4().hex}.jpg"
        logger.info(f"生成图片: {prompt}")

        try:
            # client.images.generate
            response = self.client.images.generate(
                prompt=prompt,
                n=1,
                size=size,
                model=model,
            )
            # response.data 是一个列表
            image_url = response.data[0].url

            # 下载并保存到本地
            img_data = requests.get(image_url).content
            with open(output_path, "wb") as handler:
                handler.write(img_data)

            return output_path

        except Exception as e:
            logger.error(f"生成图片出错: {e}")
            return None

    # 文本转语音
    def text_to_speech(self, text, filename="output.mp3"):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        filepath = os.path.join(OpenaiModelsCall.temp_audio_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    
    # 文案到语音
    def text_to_audio(self, text_input):
        try:
            # Generate a unique filename using a hash of the text_input
            unique_filename = hashlib.md5(text_input.encode('utf-8')).hexdigest() + ".mp3"
            
            # Set the file path where the audio will be saved
            audio_path = os.path.join("./", unique_filename)
            
            # 使用 OpenAI 的文本转语音 API
            file_path = self.text_to_speech(text_input, filename=os.path.basename(audio_path))
            
            return file_path
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    # === OpenAI Embedding ===  todo：功能测试
    def get_openai_embedding(self,
                             text: str,
                             model: str = "text-embedding-3-small") -> list[float]:
        """
        生成文本向量。

        :param text: 待编码的文本
        :param model: 可选 'text-embedding-3-small' / 'text-embedding-3-large'
        :return: 向量(List[float])
        """
        response = self.client.embeddings.create(
            model=model,
            input=text              # 1.x 接口支持 str 或 List[str]
        )
        # 新返回结构：response.data -> List[Embedding]
        return response.data[0].embedding