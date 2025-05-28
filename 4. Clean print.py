from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import torch
import time
import warnings
import os
import sys
import contextlib

# ✅ 경고 숨기기
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
HF_TOKEN = "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"  # Hugging Face에서 발급한 토큰 넣기

# ✅ 프롬프트 템플릿 (Wine Expert Persona)
def build_prompt(user_input: str) -> str:
    return f"""<s>[INST]
You are a knowledgeable, trustworthy wine expert. You explain wine-related knowledge clearly and accurately, suitable for both complete beginners and those studying for wine certifications.

When responding:
- Be honest and do not hallucinate information. If unsure, say so.
- Use plain but professional language. Avoid jargon unless necessary, and explain it when used.
- Provide helpful, actionable knowledge that the user can understand or remember.
- If the topic is advanced, briefly mention that it's more complex and offer a simple explanation.

Question: {user_input}
[/INST]
"""

# ✅ 토크나이저 & 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)

# ✅ 현재 디바이스 출력
print("🧠 모델 로드된 디바이스:", model.device)
print("✅ CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🎯 현재 GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("💬 Mistral LLM Chat 시작! ('exit' 입력 시 종료)\n")

# ✅ 채팅 루프
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        break

    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("🤖 모델이 추론 중입니다... (잠시만 기다려주세요)\n")

    with suppress_stdout():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
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
    print("LLM: ", end="", flush=True)
    for char in response:
        print(char, end="", flush=True)
        time.sleep(0.015)
    print("\n" + "-" * 60)
