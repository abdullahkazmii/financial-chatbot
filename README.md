# 💰 Financial Chatbot

A powerful AI-powered financial chatbot built using **Streamlit**, **OpenAI Assistants API**, and **Yahoo Finance**. This chatbot can provide real-time stock data, analyze market trends, generate investment insights, and even respond with audio using text-to-speech (TTS) technology.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 Features

### 🤖 AI-Powered Assistant

- Uses **GPT-4o** via **OpenAI Assistants API**
- Smart and contextual responses
- Function calling for financial tools

### 📈 Real-Time Market Data

- Stock prices using Yahoo Finance (via `yfinance`)
- Market overview for major indices: S&P 500, NASDAQ, Dow Jones
- Added support for Apple, Tesla, Google, NVIDIA, Bitcoin, etc.
- 52-week high/low, P/E ratio, volume, and more

### 🔊 Voice Response (TTS)

- Converts assistant replies to speech using OpenAI TTS
- Multiple voice options (Alloy, Nova, Shimmer, etc.)
- Optional autoplay of audio replies

### 💬 Chat Session Management

- Persistent chat session using OpenAI Threads
- All interactions saved locally during runtime
- Clear chat and thread reset options

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- OpenAI API Key ([Get yours here](https://platform.openai.com/api-keys))

### 🧰 Installation

```bash
1. **Clone the Repository**
- git clone https://github.com/yourusername/financial-chatbot.git
- cd financial-chatbot

2. **Create a Virtual Environment**
- python -m venv venv
# Windows
- venv\Scripts\activate
# macOS/Linux
- source venv/bin/activate

3. **Install Dependencies**
- pip install -r requirements.txt

4. **Run the App**
- streamlit run main.py

5. **Visit in Your Browser**
- http://localhost:8501

```

### 📁 Project Structure

financial-chatbot/
   ├── app.py               # Main Streamlit application
   ├── requirements.txt     # Python dependencies
   └── README.md            # Project documentation (this file)
