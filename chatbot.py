import asyncio
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
import torch
from dotenv import load_dotenv
from typing import AsyncGenerator

from schemas import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    DeltaMessage
)

load_dotenv()
# 모델 및 토큰
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
HF_TOKEN = os.environ.get("TOKEN")

# # 모델 초기화 변수
# chat_chain = None
old_memory = None
def init_chatbot(model_id=model_id, hf_token=HF_TOKEN):
    # chat_chain = None

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
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

    # 대화 요약! 메모리 + 체인 구성
    memory = ConversationSummaryMemory(
    llm=llm,                   # 요약에 사용할 모델
    return_messages=False,     # True면 chat format으로 저장됨
    memory_key="history",       # prompt에서 사용된 변수명과 동일해야 함
    )
    
    memory.buffer = old_memory
    # chat_chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)
    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

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


async def chain_stream_generator(chain: ConversationChain, user_input: str, model_name: str) -> AsyncGenerator[str, None]:
    global old_memory

    # 역할 설정 스트림 헤더
    first_chunk_data = ChatCompletionStreamResponse(
        model=model_name,
        choices=[ChatCompletionStreamResponseChoice(delta=DeltaMessage(role="assistant"))]
    )
    yield f"data: {first_chunk_data.model_dump_json()}\n\n"

    # # 추론 중입니다
    # thinking_message = ChatCompletionStreamResponse(
    # model=model_name,
    # choices=[ChatCompletionStreamResponseChoice(
    #     delta=DeltaMessage(content="(Thinking...)\n\n"))]
    # )
    # yield f"data: {thinking_message.model_dump_json()}\n\n"

    buffer = ""
    triggered = False
    final_response = ""

    for chunk in chain.stream({"input": user_input}):
        response_piece = chunk.get("response")
        if not response_piece:
            continue

        # 아직 "Wine Expert:" 못 찾았으면 버퍼에 누적
        if not triggered:
            buffer += response_piece
            if "Wine Expert:" in buffer:
                triggered = True
                after = buffer.split("Wine Expert:")[-1]
                for char in after:
                    final_response += char
                    chunk_data = ChatCompletionStreamResponse(
                        model=model_name,
                        choices=[ChatCompletionStreamResponseChoice(
                            delta=DeltaMessage(content=char)
                        )]
                    )
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                    await asyncio.sleep(0.02)
            continue  # 아직 trigger 안 됐거나 방금 됐으면 다음 chunk부터
        else:
            # trigger 됐으면 이후 response도 1자씩 출력
            for char in response_piece:
                final_response += char
                chunk_data = ChatCompletionStreamResponse(
                    model=model_name,
                    choices=[ChatCompletionStreamResponseChoice(
                        delta=DeltaMessage(content=char)
                    )]
                )
                yield f"data: {chunk_data.model_dump_json()}\n\n"
                await asyncio.sleep(0.02)
                
    # 끝나기 전 문장부호 확인
    final_response = final_response.strip()
    if final_response and not final_response.endswith((".", "!", "?")):
        for dot in "...":
            chunk_data = ChatCompletionStreamResponse(
                model=model_name,
                choices=[ChatCompletionStreamResponseChoice(
                    delta=DeltaMessage(content=dot)
                )]
            )
            yield f"data: {chunk_data.model_dump_json()}\n\n"
            await asyncio.sleep(0.02)
            
    old_memory = chain.memory.buffer
    yield "data: [DONE]\n\n"
    
async def fake_stream_generator(model_name: str, user_input: str) -> AsyncGenerator[str, None]:
    """
    실제 모델을 호출하는 대신, 하드코딩된 응답을 한 글자씩 스트리밍합니다.
    """
    print(f"가짜 스트리밍 생성. 사용자 입력: ")
    
    # 첫 번째 청크는 역할(role)을 알려줍니다.
    first_chunk_data = ChatCompletionStreamResponse(
        model=model_name,
        choices=[ChatCompletionStreamResponseChoice(delta=DeltaMessage(role="assistant"))]
    )
    yield f"data: {first_chunk_data.model_dump_json()}\n\n"

    # 하드코딩된 응답 텍스트
    response_text = f"이것은 하드코딩된 테스트 응답입니다. 당신의 질문은 '{user_input}' 였습니다. 이 응답은 실제 LLM을 거치지 않았습니다."
    
    # 응답 텍스트를 한 글자씩 스트리밍
    for char in response_text:
        chunk_data = ChatCompletionStreamResponse(
            model=model_name,
            choices=[ChatCompletionStreamResponseChoice(
                delta=DeltaMessage(content=char)
            )]
        )
        yield f"data: {chunk_data.model_dump_json()}\n\n"
        await asyncio.sleep(0.02) # 타이핑 효과를 위한 인위적인 딜레이

    # 마지막에는 스트림이 끝났음을 알림
    yield "data: [DONE]\n\n"
    print("가짜 스트리밍 종료.")