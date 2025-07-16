# Memory를 Buffer에서 Summary로 교체함.
# 혼합 방식도 고민. 근데 어차피 처음에 prompt로 시작하는데 굳이?

from colorama import init, Fore, Style
import os
import sys
import time
import torch
import warnings
import contextlib
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import logging

# ✅ LangChain 관련 import
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory # BufferMemory는 이전 대화를 모두 누적해서 토큰 폭발함
from langchain.prompts import PromptTemplate

from chatbot import HF_TOKEN, model_id

# ✅ 초기화
init(autoreset=True)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ✅ 모델 ID
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# ✅ 토크나이저 & 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

print("🧠 모델 로드된 디바이스:", model.device)
print("✅ CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🎯 현재 GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

# ✅ 텍스트 생성 파이프라인
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=32,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

# ✅ LangChain LLM 래퍼
llm = HuggingFacePipeline(pipeline=generator)

# ✅ 사용자 정의 프롬프트 (INST 구조 포함)
custom_template = """<s>[INST]
You are a knowledgeable, trustworthy wine expert. You explain wine-related knowledge clearly and accurately, suitable for both complete beginners and those studying for wine certifications.

When responding:
- Be honest and do not hallucinate information. If unsure, say so.
- Use plain but professional language. Avoid jargon unless necessary, and explain it when used.
- Provide helpful, actionable knowledge that the user can understand or remember.
- If the topic is advanced, briefly mention that it's more complex and offer a simple explanation.

Conversation so far:
{history}
Human: {input}
[/INST]
Wine Expert:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=custom_template
)

# ✅ 대화 요약! 메모리 + 체인 구성
memory = ConversationSummaryMemory(
    llm=llm,                   # 요약에 사용할 모델
    return_messages=False,     # True면 chat format으로 저장됨
    memory_key="history"       # prompt에서 사용된 변수명과 동일해야 함
)
chat_chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

# ✅ 대화 루프
print("💬 와인 전문가 LLM과의 대화를 시작합니다. ('exit' 입력 시 종료)\n")

while True:
    user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL)
    if user_input.strip().lower() in ['exit', 'quit']:
        break

    # print("🤖 모델이 추론 중입니다...\n")
    with suppress_stdout():
        # LangChain run 결과
        raw_response = chat_chain.run(user_input)

    # ✅ Wine Expert 이후 응답 부분만 추출
    if "Wine Expert:" in raw_response:
        response = raw_response.split("Wine Expert:")[-1].strip()
    else:
        response = raw_response.strip()

    # ✅ 끊긴 문장 보완
    if not response.endswith(('.', '!', '?')):
        response += "..."

    # ✅ 출력 (ChatGPT 스타일)
    print(Fore.BLUE + "LLM: " + Style.RESET_ALL, end="", flush=True)
    for char in response:
        print(char, end="", flush=True)
        time.sleep(0.015)
    print("\n" + "-" * 60)    