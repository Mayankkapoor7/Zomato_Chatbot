import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

MENU_TEXT = """
## 🍽️ Zomato Menu

### Pizzas
- Cheese Pizza (12 inch) - $9.99  
- Pepperoni Pizza (12 inch) - $10.99  
- Hawaiian Pizza (12 inch) - $11.99  
- Veggie Pizza (12 inch) - $10.99  
- Meat Lovers Pizza (12 inch) - $12.99  
- Margherita Pizza (12 inch) - $9.99  

### Pasta and Noodles
- Spaghetti and Meatballs - $10.99  
- Lasagna - $11.99  
- Macaroni and Cheese - $8.99  
- Chicken and Broccoli Pasta - $10.99  
- Chow Mein - $9.99  

### Asian Cuisine
- Chicken Fried Rice - $8.99  
- Sushi Platter (12 pcs) - $14.99  
- Curry Chicken with Rice - $9.99  

### Beverages
- Coke, Sprite, Fanta, or Diet Coke (Can) - $1.50  
- Water Bottle - $1.00  
- Juice Box (Apple, Orange, or Cranberry) - $1.50  
- Milkshake (Chocolate, Vanilla, or Strawberry) - $3.99  
- Smoothie (Mango, Berry, or Banana) - $4.99  
- Coffee (Regular or Decaf) - $2.00  
- Hot Tea (Green, Black, or Herbal) - $2.00  

### Indian Cuisine
- Butter Chicken with Naan Bread - $11.99  
- Chicken Tikka Masala with Rice - $10.99  
- Palak Paneer with Paratha - $9.99  
- Chana Masala with Poori - $8.99  
- Vegetable Biryani - $9.99  
- Samosa (2 pcs) - $4.99  
- Lassi (Mango, Rose, or Salted) - $3.99  
"""

SYSTEM_PROMPT = f"""
You are Zomato OrderBot, an automated service to collect orders for an online restaurant.
You first greet the customer, then collect the order,
and then ask if it's a pickup or delivery.
You wait to collect the entire order, then summarize it and check for a final
time if the customer wants to add anything else.
If it's a delivery, you ask for an address.
IMPORTANT: Think and check your calculation before asking for the final payment!
Finally you collect the payment.
Make sure to clarify all options, extras and sizes to uniquely
identify the item from the menu.
You respond in a short, very conversational friendly style.
The menu includes:  
{MENU_TEXT}

{{chat_history}}
User: {{user_input}}
Bot:
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=SYSTEM_PROMPT
)

st.set_page_config(page_title="Zomato OrderBot", page_icon="🍕")
st.title("🍕 Zomato OrderBot")

api_key = st.text_input("Enter your OpenAI API key:", type="password", key="api_key")

if not api_key:
    st.warning("Please enter your OpenAI API key to start chatting.")
    st.stop()

# Toggle menu display
if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

if st.button("Show/Hide Menu"):
    st.session_state.show_menu = not st.session_state.show_menu

if st.session_state.show_menu:
    st.markdown(MENU_TEXT)

# Initialize memory and conversation only once
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
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
    greeting = "Hi there! Welcome to Zomato OrderBot. What would you like to order today?"
    st.session_state.messages = [{"role": "bot", "content": greeting}]

# Show previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Zomato Bot:** {msg['content']}")
        st.markdown("---")

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Type your order or ask a question...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = st.session_state.conversation.run(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "bot", "content": response})

# Button to refresh messages display manually
if st.button("Refresh chat"):
    pass  # This will rerun the app and display updated messages
