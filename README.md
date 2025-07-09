 Zomato Chatbot – Intelligent Restaurant Recommendation & Food Ordering System
The Zomato Chatbot is a personalized restaurant recommendation and ordering assistant built using LangChain, GPT-4, and OpenAI embeddings, demonstrating advanced Retrieval-Augmented Generation (RAG) capabilities for structured PDF-based knowledge integration.

Problem Statement
The project aims to replicate a realistic, conversational food ordering experience where a user can:

Discover restaurants based on personal preferences (e.g., cuisine, location, rating).

Inquire about specific items, delivery ETA, or other custom preferences.

Place and manage food orders through an intelligent, context-aware chatbot.

🔧 Technical Implementation
Knowledge Base Integration (RAG)

The structured restaurant dataset (menu, ratings, delivery info) was stored in a PDF format.

Used LangChain’s PDFLoader and OpenAI embeddings to process the document.

Indexed it using FAISS (Facebook AI Similarity Search) to support fast, semantic query-based retrieval.

Applied RAG architecture: the chatbot first searches for relevant chunks in the vector store and then formulates answers using GPT-4.

Conversational Intelligence with LangChain + GPT-4

Integrated GPT-4 via LangChain’s LLMChain, enabling dynamic and coherent conversations.

Used prompt templating and memory modules to manage context over multi-turn conversations.

Enabled filtering based on multiple criteria: location, cuisine, price, delivery ETA, rating.

Streamlit Frontend with State Management

Built a responsive and interactive UI in Streamlit.

Enabled cart state tracking: users could add/remove items, view cart, and confirm orders.

Supported order customization, e.g., "add extra cheese" or "no onions" for specific items.

Persisted conversation history, cart contents, and previous orders across user sessions.

Real-Time Reasoning and Transparency

Integrated StreamlitCallbackHandler to visualize GPT-4’s reasoning steps live in the interface.

Provided explainability and increased user trust by showing how answers were derived from retrieved data.

Key Features
Smart Recommendations: Context-aware filtering of restaurants and dishes based on user intent.

PDF-Based RAG: Enabled dynamic retrieval from a custom knowledge base with semantic similarity.

End-to-End Ordering: Simulated real-world order flow from selection to cart management and confirmation.

Explainability: Real-time transparency of model decisions using LangChain tools.

Outcome & Impact
Delivered a fully functional prototype demonstrating advanced LLM capabilities in a practical food-tech scenario.

Showcased custom RAG pipelines, vector search, streamlined user experience, and multi-turn conversation management.

Could serve as a foundation for commercial-grade AI chat interfaces in food delivery platforms.

