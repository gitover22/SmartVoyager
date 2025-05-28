# 🚀 SmartVoyager

A smart and scalable AI-driven travel assistant powered by LLMs.


## 📌 Introduction

**SmartVoyager** aims to deliver an intelligent and user-friendly travel assistant system. It combines mutil APIs, and AI-generated suggestions to provide enhanced travel experiences.


## ✨ Features

* Intelligent travel planning
* openai models support
* intergate MCP & A2A & RAG & long text memory 

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:gitover22/SmartVoyager.git
cd SmartVoyager
```

### 2. Set Up Environment

Create and activate the virtual environment:

```bash
conda create -n SmartVoyager python=3.10.0 -y
conda activate SmartVoyager
```

Install required dependencies:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Edit the file `drive_environment.sh` to include your API keys:

```bash
# Example:
export MAP_API_KEY=your_api_key_here
```

Then source the file:

```bash
source ./drive_environment.sh
```

### 4. Run the Application

```bash
python entrance.py
```



## 📰 News

* 5/28 星火开发套件，接口转化为openai套件.


## ✅ Todo List

* [ ] 添加 map API 支持
* [ ] 增加MCP协议，A2A协议，重构RAG
* [ ] 重构 Web 前端界面

---

## 📬 Contributing

Feel free to fork, submit issues, or contribute via pull requests!

