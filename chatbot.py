# Steamlit에서 불러다 쓸 수 있게 변경.
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
import torch

# 모델 초기화 변수
chat_chain = None
HF_TOKEN = "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"

def init_chatbot(model_id="mistralai/Mistral-7B-Instruct-v0.1", hf_token=HF_TOKEN):
    global chat_chain

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True     # 메타 장치 에러 회피
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=generator)

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

def get_response(chatbot, user_input):
    global chat_chain
    if chat_chain is None:
        raise ValueError("Chatbot has not been initialized. Call init_chatbot() first.")

    raw_response = chat_chain.run(user_input)

    if "Wine Expert:" in raw_response:
        response = raw_response.split("Wine Expert:")[-1].strip()
    else:
        response = raw_response.strip()

    if not response.endswith((".", "!", "?")):
        response += "..."

    return response
