import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from datetime import datetime
from collections import defaultdict
import re

# ------------------ MENU ------------------
MENU_ITEMS = {
    "cheese pizza": 9.99,
    "pepperoni pizza": 10.99,
    "hawaiian pizza": 11.99,
    "butter chicken": 11.99,
    "veg biryani": 9.99,
    "gulab jamun": 4.99,
    "mango lassi": 3.99,
    "coke": 1.50
}

MENU_TEXT = """
## 🍽️ Zomato Menu

### Pizzas
- Cheese Pizza (12 inch) - $9.99  
- Pepperoni Pizza (12 inch) - $10.99  
- Hawaiian Pizza (12 inch) - $11.99  

### Indian Cuisine
- Butter Chicken - $11.99  
- Veg Biryani - $9.99  

### Desserts
- Gulab Jamun (2 pcs) - $4.99  

### Beverages
- Mango Lassi - $3.99  
- Coke (Can) - $1.50  
"""

# ------------------ SYSTEM PROMPT ------------------
SYSTEM_PROMPT = f"""
You are ZomatoBot, a friendly restaurant assistant. You help customers order food, answer questions, and address complaints in a conversational style.

The menu includes:
{MENU_TEXT}

Conversation history:
{{chat_history}}
User: {{user_input}}
Bot:
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=SYSTEM_PROMPT
)

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Zomato Chatbot", layout="wide", page_icon="🍕")
st.title("🍕 Zomato Chatbot")

# ------------------ API KEY ------------------
api_key = st.text_input("Enter your OpenAI API key:", type="password", key="api_key")
if not api_key:
    st.warning("Please enter your OpenAI API key to start chatting.")
    st.stop()

# ------------------ RIGHT SIDEBAR (Menu + Past Orders Toggle) ------------------
with st.sidebar:
    st.subheader("📋 Options")
    if "show_menu" not in st.session_state:
        st.session_state.show_menu = False
    if "show_orders" not in st.session_state:
        st.session_state.show_orders = False

    if st.button("🍽️ Show/Hide Menu"):
        st.session_state.show_menu = not st.session_state.show_menu
    if st.button("🧾 Show/Hide Past Orders"):
        st.session_state.show_orders = not st.session_state.show_orders

    if st.session_state.show_menu:
        st.markdown(MENU_TEXT)

    if st.session_state.show_orders:
        st.subheader("Your Past Orders")
        past_orders = st.session_state.get("past_orders", [])
        if not past_orders:
            st.info("No past orders yet.")
        else:
            for order in past_orders:
                st.markdown(f"🗓️ **Date:** {order['date']}")
                for item, qty in order["items"].items():
                    st.markdown(f"- {item.title()} x{qty}")
                st.markdown(f"💰 **Total:** ${order['total']:.2f}")
                st.markdown("---")

# ------------------ SESSION STATE INIT ------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "conversation" not in st.session_state:
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key
    )
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=prompt_template,
        verbose=False,
        input_key="user_input"
    )
if "messages" not in st.session_state:
    greeting = "Hi there! Welcome to Zomato Chatbot. How can I assist you today?"
    st.session_state.messages = [{"role": "bot", "content": greeting}]
    st.session_state.memory.chat_memory.add_ai_message(greeting)

# ------------------ ORDER PARSER ------------------
def parse_order(text):
    text = text.lower()
    items = defaultdict(int)
    total_cost = 0.0

    for item in MENU_ITEMS:
        matches = re.findall(rf"(\d+)?\s*{re.escape(item)}", text)
        for match in matches:
            qty = int(match) if match else 1
            items[item] += qty
            total_cost += qty * MENU_ITEMS[item]

    return dict(items), round(total_cost, 2)

# ------------------ DISPLAY MESSAGES ------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**ZomatoBot:** {msg['content']}")
        st.markdown("---")

# ------------------ CHAT FORM ------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.memory.chat_memory.add_user_message(user_input)

    # Parse orders
    items, total = parse_order(user_input)
    if items:
        order = {
            "items": items,
            "total": total,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        if "past_orders" not in st.session_state:
            st.session_state.past_orders = []
        st.session_state.past_orders.append(order)

    # Chat response
    response = st.session_state.conversation.run(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})
    st.session_state.memory.chat_memory.add_ai_message(response)

    st.rerun()
