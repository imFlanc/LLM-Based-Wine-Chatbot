# Context를 몇 턴이라도 할 수 있도록 multi-turn으로 변경
# Mistral은 애초에 한 턴 QA에 최적화된 모델이다? 결론: 지금 상태로 안 됨.

from colorama import init, Fore, Style
import os
import sys
import time
import torch
import warnings
import contextlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from chatbot import HF_TOKEN, model_id

# ✅ 초기화: 컬러, 경고, 로깅
init(autoreset=True)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ✅ 표준 출력 숨기기 도우미
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

# ✅ 프롬프트 템플릿 (Wine Expert Persona)
def build_prompt(context: list, user_input: str) -> str:
    prompt = f"""<s>[INST]
    You are a knowledgeable, trustworthy wine expert. You explain wine-related knowledge clearly and accurately, suitable for both complete beginners and those studying for wine certifications.

    When responding:
    - Be honest and do not hallucinate information. If unsure, say so.
    - Use plain but professional language. Avoid jargon unless necessary, and explain it when used.
    - Provide helpful, actionable knowledge that the user can understand or remember.
    - If the topic is advanced, briefly mention that it's more complex and offer a simple explanation.

    Question: {user_input}
    [/INST]
    """
    return prompt

# ✅ 토크나이저 & 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

# ✅ 현재 디바이스 출력
print("🧠 모델 로드된 디바이스:", model.device)
print("✅ CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🎯 현재 GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("💬 Mistral LLM Chat 시작! ('exit' 입력 시 종료)\n")

# ✅ 대화 루프
chat_history = []
while True:
    user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL)
    if user_input.strip().lower() in ['exit', 'quit']:
        break

    prompt = build_prompt(chat_history, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("🤖 모델이 추론 중입니다... (잠시만 기다려주세요)\n")

    with suppress_stdout():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # ✅ 프롬프트 이후 토큰만 추출
    output_tokens = outputs[0][len(inputs["input_ids"][0]):]
    token_list = tokenizer.convert_ids_to_tokens(output_tokens)
    response = tokenizer.convert_tokens_to_string(token_list).strip()

    # ✅ 끊긴 문장 보완
    if not response.endswith(('.', '!', '?')):
        response += "..."

    # ✅ ChatGPT 스타일 출력
    print(Fore.BLUE + "LLM: " + Style.RESET_ALL, end="", flush=True)
    for char in response:
        print(char, end="", flush=True)
        time.sleep(0.015)
    print("\n" + "-" * 60)
    
    # 대화 이력 저장 (사용자 질문 + 응답)
    chat_history.append(f"[INST] {user_input} [/INST] {response}")