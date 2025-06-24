import streamlit as st
from chatbot import init_chatbot, get_response
import time

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = init_chatbot()

# 화면 구성
st.set_page_config(page_title="Wine Expert Chatbot", page_icon="🍇")
st.title("🍇 Wine Expert Chatbot")
st.markdown("와인 도메인에 특화된 LLM이 전문가처럼 대답해드립니다.")
st.markdown("Contact us by Instargram: @c.h._min")

# 입력창
user_input = st.chat_input("무엇이든 물어보세요")

# 채팅 출력 (기존 기록만 먼저 출력)
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

if user_input:
    # 1) 사용자 메시지 바로 채팅창에 추가
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # 2) LLM 추론 전, "bot" 채팅창에 LLM 메시지 준비
    with st.chat_message("bot"):
        with st.spinner("LLM이 답변 중입니다..."):
            response = get_response(st.session_state.chatbot, user_input)
            # st.markdown(response)
            # st.session_state.chat_history.append(("bot", response))

        # 한 글자씩 출력
        placeholder = st.empty()
        displayed_text = ""
        for char in response:
            displayed_text += char
            placeholder.markdown(displayed_text)
            time.sleep(0.015)

        st.session_state.chat_history.append(("bot", response))