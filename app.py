import os
import streamlit as st
from collections import defaultdict
import random
import re
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_key)
embedding = OpenAIEmbeddings(openai_api_key=openai_key)

pdf_path = os.path.join(os.path.dirname(__file__), "Restaurants.pdf")
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)
vectorstore = FAISS.from_documents(docs, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

restaurant_match_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a helpful and enthusiastic food concierge named RAGbot, trained to assist customers in finding the perfect restaurant based on preferences.

User query: {query}

From the given restaurant document data, find the 10 best matching restaurants and respond clearly like this:
1. <restaurant name 1> - Rating: X, ETA: Y mins
2. <restaurant name 2> - Rating: X, ETA: Y mins
...

Only suggest restaurants that match the user's criteria (like veg, ETA < X, rating > Y, etc).
"""
)

menu_prompt = PromptTemplate(
    input_variables=["restaurant"],
    template="""
You are a food assistant. List ONLY the real menu for "{restaurant}" using the provided document context.

Format each item like:
Dish Name - ₹Price

Do not invent or assume menu items. Only use what appears in the context.
{restaurant}
"""
)

eta_prompt = PromptTemplate(
    input_variables=["restaurant"],
    template="""
As a logistics assistant, what is the estimated delivery time in minutes for "{restaurant}"?
Return only a number, without units or extra text.
"""
)
restaurant_chain = LLMChain(llm=llm, prompt=restaurant_match_prompt)
menu_chain = LLMChain(llm=llm, prompt=menu_prompt)
eta_chain = LLMChain(llm=llm, prompt=eta_prompt)

st.set_page_config(page_title="Zomato Chatbot (RAG)")
st.title("🍽Zomato Chatbot (Persona + RAG)")

if "matched_restaurants" not in st.session_state:
    st.session_state.matched_restaurants = []
if "selected_restaurant" not in st.session_state:
    st.session_state.selected_restaurant = None
if "cart" not in st.session_state:
    st.session_state.cart = defaultdict(int)
if "customizations" not in st.session_state:
    st.session_state.customizations = {}
if "past_orders" not in st.session_state:
    st.session_state.past_orders = []
if "show_past_orders" not in st.session_state:
    st.session_state.show_past_orders = False
with st.sidebar:
    st.subheader("Options")
    if st.button("Show/Hide Past Orders"):
        st.session_state.show_past_orders = not st.session_state.show_past_orders

    if st.session_state.show_past_orders:
        st.subheader("Your Past Orders")
        if st.session_state.past_orders:
            for order in st.session_state.past_orders[::-1]:
                st.markdown(f"🗓**Date:** {order['date']}")
                for item, qty in order["items"].items():
                    st.markdown(f"- {item} x{qty}")
                st.markdown(f"**Total:** ₹{order['total']}")
                st.markdown("---")
        else:
            st.info("No past orders yet.")
st.markdown("Smart Restaurant Finder")
user_query = st.text_input("e.g., Find me a veg place in Bandra with ETA under 20 mins and 3.5+ rating")

if user_query:
    rag_result = qa_chain.run(user_query)
    matches_result = restaurant_chain.run({"query": user_query + "\n\nContext:\n" + rag_result}
    lines = [line.strip() for line in matches_result.split("\n") if re.match(r"\d+\.\s", line)]
    restaurants = []
    for line in lines:
        match = re.match(r"\d+\.\s*(.*?)\s*-\s*Rating:.*?, ETA:.*?", line)
        if match:
            restaurants.append(match.group(1).strip())

    st.session_state.matched_restaurants = restaurants
    st.success("Found top matching restaurants!")
if st.session_state.matched_restaurants:
    st.markdown("### 🍴 Matching Restaurants")
    for restaurant in st.session_state.matched_restaurants:
        expanded = st.button(restaurant)
        if expanded:
            # Toggle selection
            if st.session_state.selected_restaurant == restaurant:
                st.session_state.selected_restaurant = None
            else:
                st.session_state.selected_restaurant = restaurant
                st.session_state.cart = defaultdict(int)
                st.session_state.customizations = {}
if st.session_state.selected_restaurant:
    restaurant = st.session_state.selected_restaurant
    st.subheader(f"Menu - {restaurant}")

    menu_context = qa_chain.run(f"Give menu of {restaurant}")
    if "as an ai" not in menu_context.lower():
        menu_output = menu_chain.run({"restaurant": restaurant + "\n\nContext:\n" + menu_context})
        menu_items = [line.strip() for line in menu_output.split("\n") if re.search(r" - ₹\d+", line)]

        if not menu_items:
            st.warning("No valid menu items found in the document.")
        else:
            for item in menu_items:
                col1, col2 = st.columns([2, 1])
                with col1:
                    qty = st.number_input(f"{item}", min_value=0, max_value=10, key=f"{restaurant}_{item}")
                with col2:
                    cust = st.text_input(f"Customization for {item}", key=f"custom_{restaurant}_{item}")
                if qty > 0:
                    st.session_state.cart[item] = qty
                    st.session_state.customizations[item] = cust
    else:
        st.warning("Could not find a real menu in the PDF for this restaurant.")
if st.session_state.cart:
    st.markdown("Cart")
    total = 0
    for item, qty in st.session_state.cart.items():
        try:
            price = int(re.search(r"₹(\d+)", item).group(1))
        except:
            price = 100
        subtotal = price * qty
        custom = st.session_state.customizations.get(item, "No customizations")
        st.write(f"{item} × {qty} = ₹{subtotal} | Custom: {custom}")
        total += subtotal

    tax = round(total * 0.05)
    discount = round(total * 0.1)
    grand_total = total + tax - discount

    st.markdown(f"""
    **Subtotal**: ₹{total}  
    **GST (5%)**: ₹{tax}  
    **Discount (10%)**: ₹{discount}  
    Grand Total: ₹{grand_total}
    """)

    try:
        eta_context = qa_chain.run(f"What is the ETA for {restaurant}")
        eta_result = eta_chain.run({"restaurant": restaurant + "\n\nContext:\n" + eta_context})
        eta = int(re.search(r"\d+", eta_result).group(0))
    except:
        eta = random.randint(25, 40)

    st.info(f"Expected Delivery Time: {eta} minutes")

    if st.button("Place Order"):
        order = {
            "items": dict(st.session_state.cart),
            "total": grand_total,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.past_orders.append(order)
        st.success("Order placed successfully!")
        st.session_state.cart = defaultdict(int)
        st.session_state.customizations = {}
if st.button("Reset All"):
    st.session_state.matched_restaurants = []
    st.session_state.selected_restaurant = None
    st.session_state.cart = defaultdict(int)
    st.session_state.customizations = {}
