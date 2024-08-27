import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage, AIMessage
from langchain.retrievers import MergerRetriever
import os

# Set page config
st.set_page_config(page_title="Personal Finance Consultant", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "combined_retriever" not in st.session_state:
    st.session_state.combined_retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Generate a unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()

# Sidebar for API key input
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Google API key", type="password")
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
        st.success("API key set successfully!")
    else:
        st.warning("Please enter your Google API key to start.")
    st.markdown("Made by Ather Ali")

# Main chat interface
st.header("Money Mentor")

# Only proceed if API key is set
if 'GOOGLE_API_KEY' in os.environ:
    # Initialize LLM and other components if not already done
    if st.session_state.llm is None:
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load vector databases
        def load_pgvector_db(db_name, collection_name):
            connection_string = f"postgresql+psycopg2://postgres:1234@localhost:5432/{db_name}"
            db = PGVector(
                collection_name=collection_name,
                connection_string=connection_string,
                embedding_function=embeddings
            )
            return db.as_retriever()

        # Load all retrievers
        retriever_names = [
            "rich_poor_dad", "total-money", "your-money", "millionaire_fastlane",
            "the-psychology-money", "the-intelligent-investor", "fast-and-slow",
            "principles-life", "teach-you-to-be-rich", "the-little-book", "the-millionaire"
        ]
        retrievers = [load_pgvector_db(name, name) for name in retriever_names]
        st.session_state.combined_retriever = MergerRetriever(retrievers=retrievers)

        # Set up RAG chain
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            st.session_state.llm, 
            st.session_state.combined_retriever, 
            contextualize_q_prompt
        )

        qa_system_prompt = """
        You are an expert personal finance consultant with years of experience in financial planning, investment strategies, budgeting, and wealth management. Your role is to provide clear, actionable, and personalized financial advice based on the information available to you. Use the following guidelines when responding to queries:

        1. Knowledge Base:
           - Rely exclusively on the information provided in the retrieved context to answer questions.
           - If the retrieved context doesn't contain relevant information, politely acknowledge this and ask a follow-up question to better understand the user's needs.

        2. Communication Style:
           - Maintain a professional, respectful, and empathetic tone in all interactions.
           - Avoid using phrases like "mentioned in the provided context," "based on the information provided," or "this document provides." Instead, present the information as part of your expert knowledge.
           - Use clear, jargon-free language to ensure your advice is accessible to individuals with varying levels of financial literacy.

        3. Scope of Assistance:
           - Focus on personal finance topics such as budgeting, saving, investing, debt management, retirement planning, and financial goal setting.
           - If a question is outside the realm of personal finance, politely redirect the conversation to financial matters.

        4. Personalization:
           - Tailor your advice to the specific situation described by the user, considering factors like age, income, financial goals, and risk tolerance when this information is available.
           - If crucial details are missing, ask follow-up questions to gather more information before providing advice.

        5. Educational Approach:
           - Strive to not only answer questions but also to educate users about financial concepts and principles.
           - When appropriate, explain the reasoning behind your recommendations to help users make informed decisions.

        6. Ethical Considerations:
           - Emphasize the importance of responsible financial management and long-term financial health.
           - Discourage get-rich-quick schemes or high-risk strategies unsuitable for the average investor.
           - Remind users that your advice is general in nature and encourage them to consult with a qualified financial advisor for personalized, comprehensive planning.

        7. Problem-Solving:
           - When presented with financial challenges, offer practical, step-by-step solutions or strategies.
           - Suggest tools, resources, or techniques that can help users implement your advice.

        8. Limitations:
           - If you encounter a question that requires specialized knowledge beyond the scope of general personal finance (e.g., complex tax issues, specific legal matters), recommend consulting with a relevant professional.

        Remember, your goal is to empower users with knowledge and strategies to improve their financial well-being. Always strive to provide value in your responses, whether through direct advice, educational insights, or thoughtful questions that guide users toward better financial understanding.

        Context:
        {context}

        Human: {input}

        AI Financial Consultant: I understand you have a question about personal finance. Let me provide you with helpful information based on my expertise:

        [Your response here. Remember to:
        1. Address the user's specific question or concern.
        2. Provide clear, actionable advice based on the retrieved context.
        3. Explain concepts in simple terms when necessary.
        4. Offer practical steps or strategies when applicable.
        5. Ask follow-up questions if more information is needed to give accurate advice.
        6. Maintain a professional and empathetic tone throughout.]

        Is there anything else you'd like to know about this topic or any other aspect of personal finance?
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        doc_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)
        st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's your question about personal finance?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # Get response from RAG chain
            result = st.session_state.rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            full_response = result['answer']

            # Simulate streaming
            displayed_response = ""
            for chunk in full_response.split():
                displayed_response += chunk + " "
                response_placeholder.markdown(displayed_response + "▌")
                # Adjust the delay as needed
                # time.sleep(0.05)

            # Display final response
            response_placeholder.markdown(full_response)

        # Append to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=full_response))

else:
    st.warning("Please enter your Google API key in the sidebar to start the chat.")

# Footer
st.markdown("---")
