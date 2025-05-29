import streamlit as st
from chatbot import init_chatbot, get_response

# ✅ 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = init_chatbot()

# ✅ 화면 구성
st.set_page_config(page_title="🍇 Wine Expert Chatbot", page_icon="🍇")
st.title("🍇 와인 전문가 채팅")
st.markdown("와인에 대해 궁금한 것을 물어보세요. 전문가처럼 대답해드림니다!")

# ✅ 입력창
user_input = st.chat_input("질문을 입력하세요")
if user_input:
    # LangChain을 통한 응답 생성
    response = get_response(st.session_state.chatbot, user_input)

    # 기록 저장
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# ✅ 채팅 출력
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
