# chat_mistral.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from chatbot import HF_TOKEN, model_id

# ✅ 모델 ID
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

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
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",         # GPU 자동 설정
    torch_dtype=torch.float16, # 4070Ti에 최적
    token=HF_TOKEN    # 로그인 인증 토큰 사용
)

# ✅ 채팅 루프 시작
print("💬 Mistral LLM Chat 시작! ('exit' 입력 시 종료)")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("🤖 모델이 추론 중입니다... (잠시만 기다려주세요)\n")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 프롬프트 제거
    if prompt in full_output:
        response = full_output.replace(prompt, "").strip()
    else:
        response = full_output.strip()

    # ✅ 토큰 단위로 출력 (ChatGPT 스타일)
    print("LLM: ", end="", flush=True)
    for token_id in outputs[0][len(inputs["input_ids"][0]):]:  # 입력 이후의 토큰만 출력
        token_str = tokenizer.decode(token_id, skip_special_tokens=True)
        print(token_str, end="", flush=True)
        time.sleep(0.015)  # 너무 빠르면 0.01~0.02 사이로 조정
    print("\n" + "-" * 60)
