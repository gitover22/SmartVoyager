import os
import gradio as gr
import requests
from gradio.components import HTML
import uuid
import base64
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import time
import json
import numpy as np
from loguru import logger
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
from pydub import AudioSegment
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from modelCall import OpenaiModelsCall
import torch

tokenizer = AutoTokenizer.from_pretrained("/disk0/home/zouguoqiang/SmartVoyager/model/rerank_model/bge-reranker-large")

style_options = ["朋友圈", "小红书", "微博", "抖音"]
openai_client = OpenaiModelsCall()

# 保存图片并获取临时路径
def save_and_get_temp_url(image):
    if not os.path.exists(OpenaiModelsCall.temp_image_dir):
        os.makedirs(OpenaiModelsCall.temp_image_dir)
    unique_filename = str(uuid.uuid4()) + ".png"
    temp_filepath = os.path.join(OpenaiModelsCall.temp_image_dir, unique_filename)
    image.save(temp_filepath)
    return temp_filepath

# 根据图像生成文本
def generate_text_from_image(image, style):
    temp_image_path = save_and_get_temp_url(image)
    prompt = "请理解这张图片"
    image_description = openai_client.image_understanding(prompt=prompt, img_path=temp_image_path)
    question = f"根据图片描述：{image_description}, 用{style}风格生成一段文字。"
    response = openai_client.chat_with_openai(question)
    return response

# 第一阶段：用户上传图片并选择风格后，点击生成文案
def on_generate_click(image, style):
    generated_text = generate_text_from_image(image, style)
    return generated_text

# 第二阶段：点击“将文案转为语音”按钮，生成并播放语音
def on_convert_click(text_output):
    return openai_client.text_to_audio(text_output)

#音频处理函数
def process_audio_file(audio_path):
    audio_segment = AudioSegment.from_file(audio_path)
    audio_segment = audio_segment.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    unique_filename = 'audio' + ".mp3"
    temp_filepath = os.path.join(OpenaiModelsCall.temp_audio_dir, unique_filename)
    audio_segment.export(temp_filepath, format="mp3")
    return temp_filepath

def process_audio(audio, history):
    print(f"接收到的音频: {audio}, 类型: {type(audio)}")

    if audio is None:
        return "没有接收到音频文件，请上传一个音频文件。", history

    if isinstance(audio, str) and os.path.isfile(audio):
        audio_path = process_audio_file(audio)
        print(f"处理的音频文件路径: {audio_path}")

        try:
            # 使用 OpenAI Whisper 模型转文本
            with open(audio_path, "rb") as audio_file:
                audio_text = openai_client.speech_to_text(audio_path=audio_file)

            print(f"语音识别结果：{audio_text}")

            if not audio_text.strip():
                return "未识别到语音，请重试。", history

            response = openai_client.chat_with_openai(audio_text)
            print(f"生成的响应: {response}")

            # 更新对话历史
            history.append((audio_text, response))
            return history

        except Exception as e:
            return f"处理音频时发生错误: {str(e)}", history

    return "无效的音频文件，请上传有效的音频。", history

rerank_path = './model/rerank_model'
rerank_model_name = 'BAAI/bge-reranker-large'


# 使用dataset/pdfs来实现知识库RAG
def extract_cities_from_text(text):
    # 简单示例：用正则匹配常见城市名
    city_list = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "西安", "重庆", "合肥", "郑州", "大连", "厦门", "黄山", "青岛", "苏州", "武汉", "天津", "丽江", "三亚"]
    found = [city for city in city_list if city in text]
    return found

# 使用cities城市名索引相关的pdf文件
def find_pdfs_with_city(cities, pdf_directory):
    matched_pdfs = {}
    for city in cities:
        matched_pdfs[city] = []
        for root, _, files in os.walk(pdf_directory):
            for file in files:
                if file.endswith(".pdf") and city in file:
                    matched_pdfs[city].append(os.path.join(root, file))
    return matched_pdfs

def get_embedding_pdf(text, pdf_directory):
    # 从文本中提取城市名称
    cities = extract_cities_from_text(text)
    # 根据城市名称匹配PDF文件
    city_to_pdfs = find_pdfs_with_city(cities, pdf_directory)
    return city_to_pdfs

def rerank(model, query, contexts, select_num):
    pairs = [[query, context] for context in contexts]
    scores = []
    for q, c in pairs:
        inputs = tokenizer(q, c, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()
            scores.append(score)
    sorted_indices = np.argsort(scores)[::-1]
    return [contexts[i] for i in sorted_indices[:select_num]]

def embedding_make(text_input, pdf_directory):

    city_to_pdfs = get_embedding_pdf(text_input, pdf_directory)
    city_list = []
    for city, pdfs in city_to_pdfs.items():
        for pdf in pdfs:
            city_list.append(pdf)
    
    if len(city_list) != 0:
        all_text = ''
        for city in city_list:
            file_opr = FileOperation()
            try:
                text, error = file_opr.read(city)
            except:
                continue
            all_text += text
            
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        all_text = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), all_text)

        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300) 
        docs = text_spliter.create_documents([all_text])
        splits = text_spliter.split_documents(docs)
        question=text_input
        
        retriever = BM25Retriever.from_documents(splits)
        retriever.k = 20
        bm25_result = retriever.invoke(question)

        question_vector = openai_client.get_openai_embedding(question)
        pdf_vector_list = []


        for i in range(len(bm25_result)):
            x = openai_client.get_openai_embedding(bm25_result[i].page_content)
            pdf_vector_list.append(x)
            time.sleep(0.65)

        query_embedding = np.array(question_vector).reshape(1, -1)
        pdf_vector_array = np.array(pdf_vector_list)

        similarities = cosine_similarity(query_embedding, pdf_vector_array)

        top_k = 10
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

        emb_list = []
        for idx in top_k_indices:
            all_page = splits[idx].page_content
            emb_list.append(all_page)
        reranker_model = AutoModelForSequenceClassification.from_pretrained("/disk0/home/zouguoqiang/SmartVoyager/model/rerank_model/bge-reranker-large")    
        documents = rerank(reranker_model, question, emb_list, 3)
        reranked = ''.join(documents)

        model_input = f'你是一个旅游攻略小助手，你的任务是，根据收集到的信息：\n{reranked}.\n来精准回答用户所提出的问题：{question}。'

        response = openai_client.chat_with_openai(model_input)
        return response
    else:
        return "请在输入中提及想要咨询的城市！"

def process_question(history, use_knowledge_base, question, pdf_directory='./dataset'):
    if use_knowledge_base=='是':
        response = embedding_make(question, pdf_directory)
    else:
        response = openai_client.chat_with_openai(question)
    history.append((question, response))
    return "", history

def clear_history(history):
    history.clear()
    return history

# 获取城市信息 
def get_location_data(location,api_key):  
    """  
    向 QWeather API 发送 GET 请求以获取天气数据。  
  
    :param location: 地点名称或经纬度（例如："beijing" 或 "116.405285,39.904989"）  
    :param api_key: 你的 QWeather API 密钥  
    :return: 响应的 JSON 数据  
    """  
    # 构建请求 URL  
    url = f"https://geoapi.qweather.com/v2/city/lookup?location={location}&key={api_key}"  
  
    # 发送 GET 请求  
    response = requests.get(url)  
  
    # 检查响应状态码  
    if response.status_code == 200:  
        # 返回 JSON 数据  
        return response.json()
    else:  
        # 处理错误情况  
        print(f"请求失败，状态码：{response.status_code}")  
        print(response.text)  
        return None
    
# 获取天气  
def get_weather_forecast(location_id,api_key):  
    """  
    向QWeather API发送请求以获取未来几天的天气预报。  
  
    参数:  
    - location: 地点ID或经纬度  
    - api_key: 你的QWeather API密钥  
    - duration: 预报的时长，'3d' 或 '7d'  
  
    返回:  
    - 响应的JSON内容  
    """
    
    # 构建请求的URL  
    url = f"https://devapi.qweather.com/v7/weather/3d?location={location_id}&key={api_key}"  
  
    # 发送GET请求  
    response = requests.get(url)  
  
    # 检查请求是否成功  
    if response.status_code == 200:  
        # 返回响应的JSON内容  
        return response.json()  
    else:  
        # 如果请求不成功，打印错误信息  
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")  
        return None  

amap_key = os.environ["amap_key"]

def get_completion(messages, model="gpt-4o"):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        seed=1024,  # 随机种子保持不变，temperature 和 prompt 不变的情况下，输出就会不变
        tool_choice="auto",  # 默认值，由系统自动决定，返回function call还是返回文字回复
        tools=[{
            "type": "function",
            "function": {

                "name": "get_location_coordinate",
                "description": "根据POI名称，获得POI的经纬度坐标",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "POI名称，必须是中文",
                        },
                        "city": {
                            "type": "string",
                            "description": "POI所在的城市名，必须是中文",
                        }
                    },
                    "required": ["location", "city"],
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "search_nearby_pois",
                "description": "搜索给定坐标附近的poi",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "longitude": {
                            "type": "string",
                            "description": "中心点的经度",
                        },
                        "latitude": {
                            "type": "string",
                            "description": "中心点的纬度",
                        },
                        "keyword": {
                            "type": "string",
                            "description": "目标poi的关键字",
                        }
                    },
                    "required": ["longitude", "latitude", "keyword"],
                }
            }
        }],
    )
    return response.choices[0].message

def get_location_coordinate(location, city):
    url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print(url)
    r = requests.get(url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None

def search_nearby_pois(longitude, latitude, keyword):
    url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print(url)
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离：{distance}米\n\n"
    return ans

def process_request(prompt):
    messages = [
        {"role": "system", "content": "你是一个地图通，你可以找到任何地址。"},
        {"role": "user", "content": prompt}
    ]
    response = get_completion(messages)
    if (response.content is None):  # 解决 OpenAI 的一个 400 bug
        response.content = ""
    messages.append(response)  # 把大模型的回复加入到对话中
    print("=====GPT回复=====")
    print(response)
    
    # 如果返回的是函数调用结果，则打印出来
    while (response.tool_calls is not None):
        # 1106 版新模型支持一次返回多个函数调用请求
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(args)
    
            if (tool_call.function.name == "get_location_coordinate"):
                print("Call: get_location_coordinate")
                result = get_location_coordinate(**args)
            elif (tool_call.function.name == "search_nearby_pois"):
                print("Call: search_nearby_pois")
                result = search_nearby_pois(**args)
    
            print("=====函数返回=====")
            print(result)
    
            messages.append({
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)  # 数值result 必须转成字符串
            })
    
        response = get_completion(messages)
        if (response.content is None):  # 解决 OpenAI 的一个 400 bug
            response.content = ""
        messages.append(response)  # 把大模型的回复加入到对话中
    
    print("=====最终回复=====")
    print(response.content)
    return response.content

def llm(query, history=[], user_stop_words=[]):
    try:
        response = openai_client.chat_with_openai(prompt=query,history=history)
        return response

    except Exception as e:
        return str(e)

# Travily 搜索引擎
if os.environ["TAVILY_API_KEY"]:
    tavily = TavilySearchResults(max_results=5)
    tavily.description = '这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！'
else:
    tavily = None
    print("警告: Tavily API 未配置，搜索功能不可用")

# 工具列表
tools = [tavily]

tool_names = 'or'.join([tool.name for tool in tools])
tool_descs = []
for t in tools:
    args_desc = []
    for name, info in t.args.items():
        args_desc.append({'name': name, 'description': info['description'] if 'description' in info else '', 'type': info['type']})
    args_desc = json.dumps(args_desc, ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s' % (t.name, t.description, args_desc))
tool_descs = '\n'.join(tool_descs)

prompt_tpl = '''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

def agent_execute(query, chat_history=[]):
    global tools, tool_names, tool_descs, prompt_tpl, llm
    
    agent_scratchpad = ''  # agent执行过程
    while True:
        history = '\n'.join(['Question:%s\nAnswer:%s' % (his[0], his[1]) for his in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(today=today, tool_descs=tool_descs, chat_history=history, tool_names=tool_names, query=query, agent_scratchpad=agent_scratchpad)
        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m' % prompt, flush=True)

        response = llm(prompt, user_stop_words=['Observation:'])
        print('\033[34m---LLM返回---\n%s\n---\033[34m' % response, flush=True)
        
        thought_i = response.rfind('Thought:')
        final_answer_i = response.rfind('\nFinal Answer:')
        action_i = response.rfind('\nAction:')
        action_input_i = response.rfind('\nAction Input:')
        observation_i = response.rfind('\nObservation:')
        
        if final_answer_i != -1 and thought_i < final_answer_i:
            final_answer = response[final_answer_i + len('\nFinal Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history
        
        if not (thought_i < action_i < action_input_i):
            return False, 'LLM回复格式异常', chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response + 'Observation: '
        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()
        
        the_tool = None
        for t in tools:
            if t.name == action:
                the_tool = t
                break
        if the_tool is None:
            observation = 'the tool not exist'
            agent_scratchpad = agent_scratchpad + response + observation + '\n'
            continue 
        
        try:
            action_input = json.loads(action_input)
            tool_ret = the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation = 'the tool has error:{}'.format(e)
        else:
            observation = str(tool_ret)
        agent_scratchpad = agent_scratchpad + response + observation + '\n'

def agent_execute_with_retry(query, chat_history=[], retry_times=10):
    for i in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history

def process_network(query):
    my_history = []
    success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
    return result

# 旅行规划师功能

prompt = """你现在是一位专业的旅行规划师，你的责任是根据旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，帮助我规划旅游行程并生成详细的旅行计划表。请你以表格的方式呈现结果。旅行计划表的表头请包含日期、地点、行程计划、交通方式、餐饮安排、住宿安排、费用估算、备注。所有表头都为必填项，请加深思考过程，严格遵守以下规则：

1. 日期请以DayN为格式如Day1，明确标识每天的行程。
2. 地点需要呈现当天所在城市，请根据日期、考虑地点的地理位置远近，严格且合理制定地点，确保行程顺畅。
3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序。位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多。时间需要按照上午、中午、晚上制定，并给出每一个位置所停留的时间（如上午10点-中午12点）。活动需要准确描述在位置发生的对应活动（如参观博物馆、游览公园、吃饭等），并需根据位置停留时间合理安排活动类型。
4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择，如步行、地铁、出租车、火车、飞机等不同的交通方式，并尽可能详细说明。
5. 餐饮安排需包含每餐的推荐餐厅、类型（如本地特色、快餐等）、预算范围，就近选择。
6. 住宿安排需包含每晚的推荐酒店或住宿类型（如酒店、民宿等）、地址、预估费用，就近选择。
7. 费用估算需包含每天的预估总费用，并注明各项费用的细分（如交通费、餐饮费、门票费等）。
8. 备注中需要包括对应行程计划需要考虑到的注意事项，保持多样性，涉及饮食、文化、天气、语言等方面的提醒。
9. 请特别考虑随行人数的信息，确保行程和住宿安排能满足所有随行人员的需求。
10.旅游总体费用不能超过预算。

现在请你严格遵守以上规则，根据我的旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，生成合理且详细的旅行计划表。记住你要根据我提供的旅行目的地、天数等信息以表格形式生成旅行计划表，最终答案一定是表格形式。以下是旅行的基本信息：
旅游出发地：{}，旅游目的地：{} ，天数：{}天 ，行程风格：{} ，预算：{}，随行人数：{}, 特殊偏好、要求：{}

"""
def chat(chat_destination, chat_history, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other):
    final_query = prompt.format(chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people, chat_other)
    
    chat_history.append((chat_destination, ''))

    response = openai_client.chat_with_openai(final_query)

    information = '旅游出发地：{}，旅游目的地：{} ，天数：{} ，行程风格：{} ，预算：{}，随行人数：{}'.format(
        chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people
    )
    chat_history[-1] = (information, response)

    yield '', chat_history

# =====================  LOGO BASE64  =====================
image_path = os.path.join(os.path.dirname(__file__), "smartVoyager.png")
with open(image_path, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode("utf-8")

# =====================  HERO SECTION  =====================
html_code = f"""
<section class="hero">
    <img src="data:image/png;base64,{encoded}" alt="SmartVoyager Logo" class="hero-logo" />
    <h1 class="hero-title">SmartVoyager智行<br><span class="subtitle">您的智能旅行管家！</span></h1>
</section>
"""

# =====================  GLOBAL CSS  =====================
custom_css = """
:root{
    --primary:#4F46E5;          /* indigo-600 */
    --primary-hover:#6366F1;    /* indigo-500 */
    --bg:#F9FAFB;
    --text:#111827;
}
body{
    font-family:'Inter','Noto Sans SC',sans-serif;
    background:linear-gradient(180deg,var(--bg) 0%,#E0E7FF 100%);
    margin:0;
}

/* Core wrapper width */
.gradio-container{
    max-width:1280px;
    margin:0 auto;
    padding:0 1rem 4rem;
}

/* ---------------- HERO ---------------- */
.hero{
    display:flex;
    flex-direction:column;
    align-items:center;
    gap:1rem;
    padding:40px 0 30px;
    text-align:center;
}
@media(min-width:768px){
    .hero{flex-direction:row;text-align:left;}
}
.hero-logo{
    width:160px;
    height:auto;
    border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,.08);
}
.hero-title{
    font-size:2.25rem;
    font-weight:700;
    line-height:1.2;
    color:var(--primary);
}
.subtitle{
    font-size:1.125rem;
    font-weight:500;
    color:var(--text);
}

/* ---------------- COMMON COMPONENTS ---------------- */
#button button{
    background:var(--primary);
    color:#ffffff;
    border:none;
    border-radius:8px;
    font-weight:600;
    padding:.55rem 1.3rem;
}
#button button:hover{background:var(--primary-hover);}

#chat-box{
    border:1px solid #E5E7EB;
    border-radius:12px;
}

.gr-accordion,.gr-accordion-open{
    border:1px solid #E5E7EB;
    border-radius:12px;
}
"""

# =====================  BUILD GRADIO APP  =====================
with gr.Blocks(css=custom_css) as demo:
    # ---- Hero banner ----
    gr.HTML(html_code)

    # =================  旅行规划助手  =================
    with gr.Tab("旅行规划助手"):
        with gr.Row():
            chat_departure = gr.Textbox(label="输入旅游出发地", placeholder="请你输入出发地")
            gr.Examples(["合肥", "郑州", "西安", "北京", "广州", "大连","厦门","南京", "大理", "上海","成都","黄山"], chat_departure, label='出发地示例',examples_per_page= 12)
            chat_destination = gr.Textbox(label="输入旅游目的地", placeholder="请你输入想去的地方")
            gr.Examples(["合肥", "郑州", "西安", "北京", "广州", "大连","厦门","南京", "大理", "上海","成都","黄山"], chat_destination, label='目的地示例',examples_per_page= 12)

        with gr.Accordion("个性化选择（天数，行程风格，预算，随行人数）", open=False):
            with gr.Group():
                with gr.Row():
                    chat_days = gr.Slider(minimum=1, maximum=10, step=1, value=3, label='旅游天数')
                    chat_style = gr.Radio(choices=['紧凑', '适中', '休闲'], value='适中', label='行程风格',elem_id="button")
                    chat_budget = gr.Textbox(label="输入预算(带上单位)", placeholder="请你输入预算")
                with gr.Row():   
                    chat_people = gr.Textbox(label="输入随行人数", placeholder="请你输入随行人数")
                    chat_other = gr.Textbox(label="特殊偏好、要求(可写无)", placeholder="请你特殊偏好、要求")

        llm_submit_tab = gr.Button("发送", visible=True,elem_id="button")
        chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天窗口", height=600)
        llm_submit_tab.click(fn=chat, inputs=[chat_destination, chatbot, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other], outputs=[ chat_destination,chatbot])

    # ===============  旅游问答助手  ===============
    def respond(message, chat_history, use_kb):
        return process_question(chat_history, use_kb, message)

    def clear_chat(chat_history):
        return clear_history(chat_history)    

    with gr.Tab("旅游问答助手"):
        with gr.Tab("知识库问答"):
            with gr.Row():
                with gr.Column():
                    msg = gr.Textbox(lines=2,placeholder="请输入您的问题（旅游景点、活动、餐饮、住宿、购物、推荐行程、小贴士等实用信息）",label="提供景点推荐、活动安排、餐饮、住宿、购物、行程推荐、实用小贴士等实用信息")
                    with gr.Row():
                        whether_rag = gr.Radio(choices=['是','否'], value='否', label='是否启用RAG')
                    with gr.Row():
                        submit_button = gr.Button("发送", elem_id="button")
                        clear_button = gr.Button("清除对话", elem_id="button")
                with gr.Column():
                    chatbot_qna = gr.Chatbot(label="聊天记录",height=521)
            submit_button.click(respond, [msg, chatbot_qna, whether_rag], [msg, chatbot_qna])
            clear_button.click(clear_chat, chatbot_qna, chatbot_qna)        

        # ===============  附近查询&联网搜索&天气查询  ===============
        Weather_APP_KEY = os.environ["Weather_APP_KEY"]
        def weather_process(location):
                api_key = Weather_APP_KEY  # 替换成你的API密钥  
                location_data = get_location_data(location, api_key)
                if not location_data:
                    return "无法获取城市信息，请检查您的输入。"
                location_id = location_data.get('location', [{}])[0].get('id')
                if not location_id:
                    return "无法从城市信息中获取ID。"
                weather_data = get_weather_forecast(location_id, api_key)
                if not weather_data or weather_data.get('code') != '200':
                    return "无法获取天气预报，请检查您的输入和API密钥。"
                html_content = "<table>"
                html_content += "<tr>"
                html_content += "<th>预报日期</th><th>白天天气</th><th>夜间天气</th><th>最高温度</th><th>最低温度</th><th>白天风向</th><th>白天风力等级</th><th>白天风速</th><th>夜间风向</th><th>夜间风力等级</th><th>夜间风速</th><th>总降水量</th><th>紫外线强度</th><th>相对湿度</th>"
                html_content += "</tr>"
                for day in weather_data.get('daily', []):
                    html_content += f"<tr><td>{day['fxDate']}</td><td>{day['textDay']} ({day['iconDay']})</td><td>{day['textNight']} ({day['iconNight']})</td><td>{day['tempMax']}°C</td><td>{day['tempMin']}°C</td><td>{day.get('windDirDay', '未知')}</td><td>{day.get('windScaleDay', '未知')}</td><td>{day.get('windSpeedDay', '未知')} km/h</td><td>{day.get('windDirNight', '未知')}</td><td>{day.get('windScaleNight', '未知')}</td><td>{day.get('windSpeedNight', '未知')} km/h</td><td>{day.get('precip', '未知')} mm</td><td>{day.get('uvIndex', '未知')}</td><td>{day.get('humidity', '未知')}%</td></tr>"
                html_content += "</table>"  
                return HTML(html_content)  

        def clear_history_audio(history):
            history.clear()
            return history

        def clear_chat_audio(chat_history):
            return clear_history_audio(chat_history)

        with gr.Tab("附近查询&搜索&天气"):
            with gr.Row():
                with gr.Column():
                    query_near = gr.Textbox(label="查询附近的餐饮、酒店等", placeholder="例如：合肥市高新区中国声谷产业园附近的美食")
                    result = gr.Textbox(label="查询结果", lines=2)
                    submit_btn = gr.Button("查询附近的餐饮、酒店等",elem_id="button")
                    gr.Examples(["合肥市高新区中国声谷产业园附近的美食", "北京三里屯附近的咖啡", "南京市玄武区新街口附近的甜品店", "上海浦东新区陆家嘴附近的热门餐厅", "武汉市光谷步行街附近的火锅店", "广州市天河区珠江新城附近的酒店"], query_near)
                    submit_btn.click(process_request, inputs=[query_near], outputs=[result])
                with gr.Column():
                    query_network = gr.Textbox(label="联网搜索问题", placeholder="例如：秦始皇兵马俑开放时间")
                    result_network = gr.Textbox(label="搜索结果", lines=2)
                    submit_btn_network = gr.Button("联网搜索",elem_id="button")
                    gr.Examples(["秦始皇兵马俑开放时间", "合肥有哪些美食", "北京故宫开放时间", "黄山景点介绍", "上海迪士尼门票需要多少钱"], query_network)
                    submit_btn_network.click(process_network, inputs=[query_network], outputs=[result_network])

            weather_input = gr.Textbox(label="请输入城市名查询天气", placeholder="例如：北京")
            weather_output = gr.HTML(value="", label="天气查询结果")
            query_button = gr.Button("查询天气",elem_id="button")
            query_button.click(weather_process, [weather_input], [weather_output])

    # ===============  旅行文案助手  ===============
    with gr.Tab("旅行文案助手"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图像",height= 230)                
                
            with gr.Column():    
                style_dropdown = gr.Dropdown(choices=style_options, label="选择风格模式", value="朋友圈")
            with gr.Column():
                audio_output = gr.Audio(label="音频播放", interactive=False, visible=True)
                
        with gr.Row():
            generate_button = gr.Button("生成文案", visible=True,elem_id="button")
            convert_button1 = gr.Button("文案转语音", visible=True,elem_id="button")
        with gr.Row():
            with gr.Column():
                
                generated_text = gr.Textbox(lines=7, label="生成的文案", visible=True)
                prompt_input = gr.Textbox(label="文生图输入提示", placeholder="可以把生成文案输入到这里，帮你生成图片")
                generate_btn = gr.Button("生成图片",elem_id="button") 
            with gr.Column():
                output_image = gr.Image(label="生成的图片",height= 400)     
        generate_button.click(on_generate_click, inputs=[image_input, style_dropdown], outputs=[generated_text])
       
        convert_button1.click(on_convert_click, inputs=[generated_text], outputs=[audio_output])

        generate_btn.click(openai_client.generate_image, inputs=prompt_input, outputs=output_image)

class FileOperation:
    """
    文件操作类，支持读取 PDF 文件内容。
    """
    def read(self, path):
        """
        读取 PDF 文件内容。
        :param path: PDF 文件路径
        :return: (文本内容, 错误信息)
        """
        try:
            loader = PyMuPDFLoader(path)
            docs = loader.load()
            # docs 是 Document 对象列表，拼接所有内容
            text = '\n'.join([doc.page_content for doc in docs])
            return text, None
        except Exception as e:
            return '', str(e)

if __name__ == "__main__":
    demo.queue().launch(share=True)
