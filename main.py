import streamlit as st
import openai
import yfinance as yf
import json
import time
from typing import Optional, Dict, Any

# Page Configuration
st.set_page_config(
    page_title="Financial Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "alloy"


# Financial Data Functions
@st.cache_data(ttl=300)
def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get real-time stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        hist = ticker.history(period="1d")

        if hist.empty:
            return {"error": f"No data available for {symbol}"}

        current_price = float(hist["Close"].iloc[-1])

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "company_name": info.get("longName", symbol),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
        }
    except Exception as e:
        return {"error": f"Could not fetch data for {symbol}: {str(e)}"}


@st.cache_data(ttl=300)
def get_market_overview() -> Dict[str, Any]:
    """Get major market indices overview"""
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "NVIDIA": "NVDA",
        "Russell 2000": "^RUT",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
        "S&P/TSX Composite": "^GSPTSE",
        "Euro Stoxx 50": "^STOXX50E",
        "CAC 40": "^FCHI",
        "ASX 200": "^AXJO",
        "Bovespa": "^BVSP",
        "Sensex": "^BSESN",
        "Swiss Market Index": "^SSMI",
        "KOSPI": "^KS11",
        "Nifty 50": "^NSEI",
        "Jakarta Composite": "^JKSE",
        "Straits Times": "^STI",
    }

    results = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if not hist.empty:
                current = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
                change = current - prev
                change_pct = (change / prev) * 100

                results[name] = {
                    "current": round(current, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_pct, 2),
                }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


# OpenAI Functions
def create_openai_client(api_key: str) -> openai.OpenAI:
    """Create OpenAI client"""
    return openai.OpenAI(api_key=api_key)


def create_assistant(client: openai.OpenAI) -> Optional[str]:
    """Create financial assistant"""
    try:
        assistant = client.beta.assistants.create(
            name="Financial Advisor Bot",
            instructions="""You are a knowledgeable financial advisor chatbot. You can help users with:
- Stock analysis and recommendations  
- Market trends and insights
- Investment strategies
- Financial planning advice
- Real-time market data interpretation

Always provide accurate, helpful financial information while reminding users that this is not personalized financial advice and they should consult with a licensed financial advisor for investment decisions.

When users ask about specific stocks or market data, use the provided tools to give current information.""",
            model="gpt-4o",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_stock_data",
                        "description": "Get real-time stock data for a given symbol",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Stock symbol (e.g., AAPL, MSFT)",
                                }
                            },
                            "required": ["symbol"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_market_overview",
                        "description": "Get overview of major market indices",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                },
            ],
        )
        return assistant.id
    except Exception as e:
        st.error(f"Failed to create assistant: {e}")
        return None


def text_to_speech(
    client: openai.OpenAI, text: str, voice: str = "alloy"
) -> Optional[bytes]:
    """Convert text to speech using OpenAI TTS"""
    try:
        if len(text) > 4000:
            text = text[:4000] + "..."

        response = client.audio.speech.create(
            model="tts-1", voice=voice, input=text, response_format="mp3"
        )
        return response.content
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


def handle_function_call(function_name: str, arguments: Dict[str, Any]) -> str:
    """Handle function calls from assistant"""
    try:
        if function_name == "get_stock_data":
            symbol = arguments.get("symbol", "")
            result = get_stock_data(symbol)
            return json.dumps(result)
        elif function_name == "get_market_overview":
            result = get_market_overview()
            return json.dumps(result)
        else:
            return json.dumps({"error": "Function not implemented"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def chat_with_assistant(
    client: openai.OpenAI,
    assistant_id: str,
    message: str,
    thread_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Chat with OpenAI assistant"""
    try:
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id

        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=message
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )

        while run.status in ["queued", "in_progress", "requires_action"]:
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    output = handle_function_call(function_name, arguments)

                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": output}
                    )

                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
            else:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            response = messages.data[0].content[0].text.value
            return response, thread_id
        else:
            return f"Error: Run failed with status {run.status}", thread_id

    except Exception as e:
        return f"Error: {str(e)}", thread_id


def process_user_input(prompt: str, client: openai.OpenAI):
    """Process user input and generate response"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Processing..."):
            response, thread_id = chat_with_assistant(
                client,
                st.session_state.assistant_id,
                prompt,
                st.session_state.thread_id,
            )
            st.session_state.thread_id = thread_id
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Auto-play TTS if enabled
    if st.session_state.enable_tts:
        audio_data = text_to_speech(client, response, st.session_state.selected_voice)
        if audio_data:
            st.audio(audio_data, format="audio/mp3", autoplay=True)


# Sidebar
with st.sidebar:
    st.markdown("### 💰 Financial Chatbot")
    st.markdown("*AI-powered financial advisor*")
    st.divider()

    st.markdown("#### 🔑 Configuration")
    api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key",
        placeholder="sk-...",
    )

    if api_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key
        st.session_state.assistant_id = None

    if not api_key:
        st.warning("⚠️ API key required")
    elif api_key.startswith("sk-") and len(api_key) > 20:
        st.success("✅ API key configured")

        st.divider()

        # # TTS Settings
        # st.markdown("#### 🔊 Audio Settings")

        # enable_tts = st.checkbox(
        #     "🔊 Enable Text-to-Speech",
        #     value=st.session_state.enable_tts,
        #     help="AI will speak responses aloud",
        # )
        # st.session_state.enable_tts = enable_tts

        # if enable_tts:
        #     voice_options = {
        #         "Alloy": "alloy",
        #         "Echo": "echo",
        #         "Fable": "fable",
        #         "Onyx": "onyx",
        #         "Nova": "nova",
        #         "Shimmer": "shimmer",
        #     }

        #     selected_voice_name = st.selectbox(
        #         "Voice:",
        #         options=list(voice_options.keys()),
        #         index=list(voice_options.values()).index(
        #             st.session_state.selected_voice
        #         ),
        #         help="Choose AI voice personality",
        #     )
        #     st.session_state.selected_voice = voice_options[selected_voice_name]

        # st.divider()

        # Quick Market Overview
        if st.button("📈 Market Overview", use_container_width=True):
            with st.spinner("Loading market data..."):
                market_data = get_market_overview()
                for name, data in market_data.items():
                    if "error" not in data:
                        change_emoji = "🟢" if data["change"] >= 0 else "🔴"
                        st.metric(
                            label=f"{change_emoji} {name}",
                            value=f"{data['current']:,.0f}",
                            delta=f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)",
                        )

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.rerun()
    else:
        st.error("❌ Invalid API key format")

# Main Content
if not api_key or not api_key.startswith("sk-"):
    st.title("💰 Financial Chatbot")
    st.subheader("Your AI-powered financial advisor")

    st.info("👈 **Please enter your OpenAI API key in the sidebar to get started**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌟 Features")
        st.markdown("""
        - 📈 **Real-time stock data**
        - 🤖 **AI financial advisor** 
        - 🔊 **Text-to-speech output**
        - 📊 **Market analysis**
        """)

    with col2:
        st.markdown("### 🔑 Setup Instructions")
        st.markdown("""
        1. Get API key: [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Enter it in the sidebar
        3. Start chatting!
        
        **Example Questions:**
        - "What's Apple's stock price?"
        - "Give me a market overview"
        - "Should I invest in tech stocks?"
        """)

else:
    client = create_openai_client(api_key)

    if not st.session_state.assistant_id:
        with st.spinner("🤖 Initializing Assistant..."):
            assistant_id = create_assistant(client)
            if assistant_id:
                st.session_state.assistant_id = assistant_id
                st.success("✅ Assistant ready!")
            else:
                st.error("❌ Failed to initialize assistant")
                st.stop()

    st.title("💰 Financial Chatbot")
    st.markdown("*Ask me anything about stocks, markets, or investments!*")

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Add listen button for assistant messages
            if message["role"] == "assistant":
                if st.button("🔊 Listen", key=f"tts_{i}"):
                    with st.spinner("🔊 Generating speech..."):
                        audio_data = text_to_speech(
                            client, message["content"], st.session_state.selected_voice
                        )
                        if audio_data:
                            st.audio(audio_data, format="audio/mp3")

    # Chat Input
    if prompt := st.chat_input("Ask about stocks, markets, or financial advice..."):
        process_user_input(prompt, client)
        st.rerun()

    # Quick Actions
    st.markdown("---")
    st.markdown("**💡 Quick Actions:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Market Summary", use_container_width=True):
            process_user_input("Give me today's market summary", client)
            st.rerun()

    with col2:
        if st.button("🔥 Hot Stocks", use_container_width=True):
            process_user_input("What are some trending stocks today?", client)
            st.rerun()

    with col3:
        if st.button("📊 Investment Advice", use_container_width=True):
            process_user_input(
                "Give me an investment advice with the current market trends which will benefit in coming years",
                client,
            )
            st.rerun()
