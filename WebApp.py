import streamlit as st
from Query_Script_RAG_Pipeline import retrieve_prompt_with_context
from openai import OpenAI
from Keys import KEY1


st.set_page_config(page_title= "GenAI PDF Query", page_icon="📕")
st.title("📕 GenAI PDF Study Assistant")

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=KEY1
                )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content" : ("You are a helpful assistant who talks about according to the given context, which is related to GenAI." "You clear about the problems that the student facing according to the context. ")}]

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
                prompt = retrieve_prompt_with_context(Ques)

                # Add context prompt temporarily
                messages = st.session_state.messages + [
                    {"role": "user", "content": prompt}
                ]

                chats = client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                messages= messages,
                temperature=0,
                max_tokens=400)

                reply = chats.choices[0].message.content
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"Error: {e}")