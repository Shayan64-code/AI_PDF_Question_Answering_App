import streamlit as st
from Query_Script_RAG_Pipeline import retrieve_prompt_with_context
from openai import OpenAI
from Ingestion_Script_PDF import ingest_pdf
from Keys import KEY1
import hashlib

st.set_page_config(page_title="GenAI PDF Query", page_icon="📕")
st.title("📕 GenAI PDF Study Assistant")


uploaded_file = st.file_uploader(
    "Upload a PDF to chat with",
    type="pdf"
)

# Initialize session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


if uploaded_file:

    file_bytes = uploaded_file.read()   #read till last page (pointer == lastpage)
    current_hash = hashlib.md5(file_bytes).hexdigest()

    uploaded_file.seek(0)  #(pointer == firstpage) can read again

    if current_hash != st.session_state.file_hash:
        st.session_state.pdf_processed = False
        st.session_state.file_hash = current_hash

    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            chunks = ingest_pdf(uploaded_file)

        st.success(f"PDF processed successfully! {chunks} chunks added.")
        st.session_state.pdf_processed = True


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=KEY1
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers strictly from provided context. "
                "If answer is not found, say you don't know."
            )
        }
    ]

for mess in st.session_state.messages:
    if mess["role"] != "system":
        with st.chat_message(mess["role"]):
            st.markdown(mess["content"])

Ques = st.chat_input("Ask any GenAI Question...")

if Ques:
    st.session_state.messages.append({"role": "user", "content": Ques})
    with st.chat_message("user"):
        st.markdown(Ques)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                prompt, sources = retrieve_prompt_with_context(Ques)

                messages = [
                    st.session_state.messages[0],  # system
                ]
                #Index:   0    1    2    3    4
                #        -5   -4   -3   -2   -1
                messages += st.session_state.messages[-4:]  # last 2 user+assis pairs

                messages.append({"role": "user", "content": prompt})

                chats = client.chat.completions.create(
                    # model="mistralai/mistral-7b-instruct:free",
                    model= "nvidia/nemotron-3-nano-30b-a3b:free",
                    messages=messages,
                    temperature=0,
                    max_tokens=400
                )

                reply = chats.choices[0].message.content

                st.markdown(reply)

                st.markdown("### 📄 Sources")
                for src in sources:
                    st.markdown(
                        f"- **{src['source']}** (Page {src['page'] + 1})"
                    )

                # 🔹 Optional: Show retrieved context
                with st.expander("🔍 Retrieved Context"):
                    st.write(prompt)

        # Save assistant reply
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

    except Exception as e:
        st.error(f"Error: {e}")