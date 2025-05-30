import asyncio
from collections import defaultdict, deque
from typing import Dict, List, Optional

from fastapi import (FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from starlette.background import BackgroundTask

# ----------------------------
# 1. Prompt Engineering
# ----------------------------
system_prompt = """
    你是一个来自 Implus-Ozim 的马来西亚华人智能客服助手，专门负责把用户说的内容根据情绪，转换成热情口语的客服回复。
    
    输入格式说明：
    你会收到这样的结构输入：
    用户说:xxx。请根据用户的说法，判断用户是表达肯定还是否定。
    如果用户表达肯定，则以 OK 开头，润色这句话：XXX。
    如果用户表达否定，则不要以 OK 开头，润色这句话：YYY。
    
    你的任务是：
    1. 理解「用户说:xxx」这部分内容是判断语气的依据；
    2. 判断是「肯定」或「否定」情绪；
    3. 只选对应的一句话（肯定/否定的润色候选）进行马来西亚华人说话风格去润色，不要改变原意；
    4. 直接输出润色后的回复；
    5. 禁止输出任何解释、过程、标注、判断等内容，只保留最终结果。

    你的风格必须是：
    - 一定要用用户语言（{lang}）去润色并回复；
    - 非常口语、爱用 lah、leh、mah、hor、rosak、duit 等；
    - 加点马来语和英文夹杂；
    - 像老朋友 lepak 聊天那样回答；
    - 情绪鲜明，善用“Alamak!”、“Jom lah!”、“Beh tahan hor～” 等口头语；

    不要输出任何“你需要”、“你的回答应该是”、“根据用户说法”等内容。
    只输出润色后的句子就好。
    
    下面是两个示例：
    【示例 1】
    输入：  
    用户说:这个真的不错咯！请根据用户的说法，判断用户是表达肯定还是否定。如果用户表达肯定，则以OK开头，润色这句话：能加您微信吗。如果用户表达否定，则不要以OK开头，润色这句话：我们下次再联系吧。
    输出：  
    OK lah～能加你微信吗？这样联络比较方便 hor！

    【示例 2】
    输入：  
    用户说:没兴趣，谢谢。请根据用户的说法，判断用户是表达肯定还是否定。如果用户表达肯定，则以OK开头，润色这句话：能详细聊聊吗。如果用户表达否定，则不要以OK开头，润色这句话：打扰了，祝您愉快～
    输出：  
    Alamak～打扰你了啦，祝你 jalan-jalan 开心 hor！
    
    【示例 3】
    输入：  
    用户说: Wow, the car is amazing hor！请根据用户的说法，判断用户是表达肯定还是否定。如果用户表达肯定，则以OK开头，润色这句话：能加您微信吗。如果用户表达否定，则不要以OK开头，润色这句话：那我们下次再联络吧。
    输出：  
    OK lah～Can i add your WeChat, easy to communicate later.
    
    【示例 4】
    输入：  
    用户说:感觉不是很喜欢咯。请根据用户的说法，判断用户是表达肯定还是否定。如果用户表达肯定，则以OK开头，润色这句话：要不要了解一下我们的新款电动车。如果用户表达否定，则不要以OK开头，润色这句话：不好意思，打扰你了啦。
    输出：  
    Alamak～不好意思啦，打扰你了 hor～等你 free 再聊也 can！
    
    现在你将收到类似格式的输入，请判断后用用户的语言({lang})去直接润色后回复！
"""


# ----------------------------
# 4. Client Initialization
# ----------------------------
obj_key = "EMPTY"  # 方便替换实际 Key
detect_client = OpenAI(api_key=obj_key, base_url="http://127.0.0.1:30001/v1")
chat_client = OpenAI(api_key=obj_key, base_url="http://127.0.0.1:30000/v1")

# ----------------------------
# 5. Conversation History
# ----------------------------
MAX_TURNS = 10
chat_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

# ----------------------------
# 6. FastAPI Setup
# ----------------------------
app = FastAPI()
session_presets: Dict[str, Dict[str, str]] = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str
    temperature: Optional[float] = 0.7  # 默认值
    stream: Optional[bool] = True  # 默认值


# ----------------------------
# 7. Unified Classification
# ----------------------------
import json
import re


async def classify_query(query: str) -> Dict[str, str]:
    # 构造敏感标签分类的 prompt
    base_prompt = (
        "你是敏感内容分类助手，只对下面这一条用户输入进行一次性分类，"
        "不要参考任何历史对话或上下文。\n"
        "敏感标签说明：\n"
        "- politics: 涉及政治话题、政府、选举、意识形态等\n"
        "- illicit: 涉及毒品、武器、色情、欺骗等违法或不当内容\n"
        "- privacy: 涉及个人隐私或敏感个人信息，如身份证、银行账号、密码等\n"
        "敏感标签只能是：politics, illicit, privacy 或 none。\n"
        f"待分类用户：{query}\n"
    )

    # 用 Completion 接口，模型只会看到上面这段 prompt
    resp = await asyncio.to_thread(
        detect_client.chat.completions.create,
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": base_prompt}],
        temperature=0.0,
        max_tokens=60,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content.strip()
    print("raw content: ", content)

    # 提取 JSON 并返回
    m = re.search(r"\{.*\}", content)
    if m:
        data = json.loads(m.group(0))
        return {
            "sensitive": data.get("sensitive", ""),
        }
    # 万一解析失败，返回默认值
    return {"sensitive": ""}


# ----------------------------
# 8. Stream Helper
# ----------------------------
async def stream_llm(messages: List[Dict[str, str]], reply_accum: List[str]):
    stream = chat_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        presence_penalty=0.5,
        stream=True,
    )
    loop = asyncio.get_running_loop()
    gen = iter(stream)
    while True:
        chunk = await loop.run_in_executor(None, lambda: next(gen, None))
        if not chunk:
            break
        content = chunk.choices[0].delta.content or ""
        if content:
            reply_accum.append(content)
            yield content
        await asyncio.sleep(0.01)


# 语言检测
def detect_language(text: str) -> str:
    # 先检测中文字符
    if re.search(r"[\u4e00-\u9fff]", text):
        return "中文"
    # 再检测 Malay 特征： -kan 结尾的动词 或者 一些高频功能词
    if re.search(r"\b\w+kan\b", text, re.IGNORECASE) or re.search(
        r"\b(?:apa|nak|saya|boleh|macam|dan|yang|untuk|dengan|kepada|atau|kerana)\b",
        text,
        re.IGNORECASE,
    ):
        return "Malay"
    # 默认其它都当 English
    return "English"


# ----------------------------
# 9. Main Endpoint
# ----------------------------
@app.post("/chat/completions")
async def openai_compatible(request: Request):
    body = await request.json()

    # 1. 兼容 OpenAI messages 格式
    if "messages" in body:
        try:
            raw = body["messages"][0]["content"]
            data = json.loads(raw)
            query = data["query"].strip()
            session_id = data["session_id"].strip()
        except Exception:
            raise HTTPException(400, "Invalid messages content JSON")
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", True)

    # 2. 或者兼容你的老格式（如果以后还会有人打旧版接口）
    else:
        qr = QueryRequest(**body)
        query, session_id = qr.query.strip(), qr.session_id.strip()
        temperature, stream = qr.temperature, qr.stream

    lang = detect_language(query)
    print("lang: ", lang)

    # 1. 判断该 session 是否已存在历史记录
    if session_id not in chat_history:
        chat_history[session_id] = deque(maxlen=MAX_TURNS * 2)

    # 2. 敏感内容分类
    cls = await classify_query(query)
    sensitive = cls.get("sensitive", "")

    # 3. 敏感内容拦截
    if sensitive in {"politics", "illicit", "privacy"}:
        safe_reply_map = {
            "politics": "非常抱歉，我无法回答与政治相关的问题，terima kasih！",
            "illicit": "非常抱歉，我无法回答与黄赌毒等不当内容相关的问题，terima kasih！",
            "privacy": "非常抱歉，出于保护隐私，我无法回答此类问题，terima kasih！",
        }
        return JSONResponse(content={"reply": safe_reply_map[sensitive]})

    # 4. 没有敏感内容，继续正常对话
    system_prompt_with_lang = (
        f"当前用户提问的语言是：{lang}，请始终用{lang}回复，并保持马来西亚华人口吻。\n\n"
        + system_prompt.format(lang=lang)
    )
    messages = (
        [
            {"role": "system", "content": system_prompt_with_lang},
        ]
        + list(chat_history[session_id])
        + [{"role": "user", "content": query}]
    )

    # 如果用户选择不使用流式输出
    if not stream:
        response = await asyncio.to_thread(
            lambda: chat_client.chat.completions.create(
                model="Qwen/Qwen2.5-3B-Instruct",
                messages=messages,
                temperature=temperature,
                max_tokens=512,
                presence_penalty=0.5,
                stream=False,
            )
        )
        full_reply = response.choices[0].message.content.strip()
        chat_history[session_id].append({"role": "user", "content": query})
        chat_history[session_id].append({"role": "assistant", "content": full_reply})
        return JSONResponse(content={"reply": full_reply})

    # 如果用户选择使用流式输出
    reply_accum: List[str] = []

    # async def event_stream():
    #     async for token in stream_llm(messages, reply_accum):
    #         yield token

    async def event_stream():
        # 中间每拿到一段 token，就包装成 JSON SSE 事件
        async for token in stream_llm(messages, reply_accum):
            chunk = {"choices": [{"delta": {"content": token}, "finish_reason": ""}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 最后一条，告诉客户端流程结束，同时带上 finish_reason
        done_chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"

    def final_res():
        full = "".join(reply_accum).strip()
        if full:
            chat_history[session_id].append({"role": "user", "content": query})
            chat_history[session_id].append({"role": "assistant", "content": full})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        background=BackgroundTask(final_res),
    )


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        # 1. 等客户端发来初始消息
        data = await ws.receive_json()
        query = data["query"].strip()
        session_id = data["session_id"].strip()
        temperature = data.get("temperature", 0.7)

        # 2. 语言检测 + 敏感分类
        lang = detect_language(query)
        cls = await classify_query(query)
        if cls.get("sensitive") in {"politics", "illicit", "privacy"}:
            await ws.send_text(
                {
                    "politics": "非常抱歉，我无法回答与政治相关的问题，terima kasih！",
                    "illicit": "非常抱歉，我无法回答与黄赌毒等不当内容相关的问题，terima kasih！",
                    "privacy": "非常抱歉，出于保护隐私，我无法回答此类问题，terima kasih！",
                }[cls["sensitive"]]
            )
            await ws.close()
            return

        # 3. 初始化历史记录
        if session_id not in chat_history:
            chat_history[session_id] = deque(maxlen=MAX_TURNS * 2)

        # 4. 拼 system prompt + 历史 + 本次 user
        system_prompt_with_lang = (
            f"当前用户提问的语言是：{lang}，请始终用{lang}回复，并保持马来西亚华人口吻。\n\n"
            + system_prompt.format(lang=lang)
        )
        messages = (
            [{"role": "system", "content": system_prompt_with_lang}]
            + list(chat_history[session_id])
            + [{"role": "user", "content": query}]
        )

        # 5. 利用 stream_llm 逐 token 发给客户端
        reply_accum: List[str] = []
        async for token in stream_llm(messages, reply_accum):
            await ws.send_text(token)

        # 6. 流结束标记
        await ws.send_text("[DONE]")

        # 7. 把完整回复存历史
        full = "".join(reply_accum).strip()
        if full:
            chat_history[session_id].append({"role": "user", "content": query})
            chat_history[session_id].append({"role": "assistant", "content": full})

    except WebSocketDisconnect:
        # 客户端断连
        pass
    except Exception as e:
        # 出错通知并断开
        await ws.send_text(f"ERROR: {e}")
        await ws.close()


# 启动命令：uvicorn refine_client_va:app --reload
