import asyncio
import re
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from starlette.background import BackgroundTask

# ----------------------------
# Prompt Templates
# ----------------------------
# classification_prompt_template = (
#     "不管用户用什么语言，请判断用户这句话：“{text}”表达的情绪是“肯定”还是“否定”。"
#     "仅输出一个 JSON 对象，不能有任何多余文字或说明，属性名和字符串值都必须使用双引号，"
#     "格式必须严格符合：{\"sentiment\":\"肯定\"} 或 {\"sentiment\":\"否定\"}。"
#     "“肯定”表示认可，“否定”表示不认可。"
#     "示例1 — 输入: \"i feel so bad\"，输出: {\"sentiment\":\"否定\"}"
#     "示例2 — 输入: \"tak nak lah\"，输出: {\"sentiment\":\"否定\"}"
#     "示例3 — 输入: \"不错哦\"，输出: {\"sentiment\":\"肯定\"}"
# )

# polish_prompt_affirm = (
#     '一定要以OK开头，用马来西亚华人客服口语风格润色这句话：{to_refine}，不要改变原意'
#     '你的风格必须是：用用户语言({lang})去润色{to_refine}这句话，要求非常口语、爱用 lah、leh、mah、hor、rosak、duit；'
#     '加点马来语和英文夹杂；像老朋友 lepak 聊天；情绪鲜明，善用“Alamak!”、“Jom lah!”、“Beh tahan hor～”等口头语。'
#     '务必参考以下示例句式，润色输出一句话，不要改变原意，不要多余说明。'
#     '润色模板，示例参考:\n'
#     'OK lah～能加你微信吗？方便联络一下，改天带你 jalan-jalan 试驾 hor！\n'
# )
# polish_prompt_neg = (
#     '请不要以OK开头，用马来西亚华人客服口语风格润色这句话：{to_refine}，不要改变原意'
#     '你的风格必须是：用用户语言({lang})去润色{to_refine}这句话，要求非常口语、爱用 lah、leh、mah、hor、rosak、duit；'
#     '加点马来语和英文夹杂；像老朋友 lepak 聊天；情绪鲜明，善用“Alamak!”、“Jom lah!”、“Beh tahan hor～”等口头语。'
#     '务必参考以下示例句式，润色输出一句话，不要改变原意，不要多余说明。'
#     '润色模板，示例参考:\n'
#     'Alamak～不好意思啦，打扰你了 hor～等你 free 再聊也 can！\n'
# )
classification_prompt_template = (
    "不管用户用什么语言，请判断用户这句话：“{text}”表达的情绪是“肯定”还是“否定”。"
    "仅输出一个 JSON 对象，不能有任何多余文字或说明，属性名和字符串值都必须使用双引号，"
    "格式必须严格符合：{\"sentiment\":\"肯定\"} 或 {\"sentiment\":\"否定\"}。"
    "“肯定”表示认可，“否定”表示不认可。"
    "不要过度解读用户的情绪，就从字面意思去判断"
    "示例1 — 输入: \"i feel so bad\"，输出: {\"sentiment\":\"否定\"}"
    "示例2 — 输入: \"tak nak lah\"，输出: {\"sentiment\":\"否定\"}"
    "示例3 — 输入: \"不错哦\"，输出: {\"sentiment\":\"肯定\"}"
)

polish_prompt_affirm = (
    '用({lang})去润色这句话：{to_refine}；在润色后的结果前面一定要把OK作为开头'
    # '你的回复一定要以OK开头，然后加上你润色后的话，不要改变原意，不要多余说明。'
)
polish_prompt_neg = (
    '请不要以OK开头，用({lang})去润色这句话：{to_refine}'
    # '润色输出，不要改变原意，不要多余说明。'
)

# ----------------------------
# Client Initialization
# ----------------------------
obj_key = "EMPTY"
detect_client = OpenAI(api_key=obj_key, base_url="http://127.0.0.1:30001/v1")
chat_client   = OpenAI(api_key=obj_key, base_url="http://127.0.0.1:30000/v1")

# ----------------------------
# Conversation History
# ----------------------------
MAX_TURNS = 10
chat_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: str
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True

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
# 工具函数：拆解用户输入
# ----------------------------
def parse_structured_input(query: str):
    try:
        # 提取用户原话
        user_said = query.split("用户说:", 1)[1].split("。请", 1)[0].strip()
        # 定义标记
        marker_affirm = "如果用户表达肯定"
        marker_neg = "如果用户表达否定"
        tag = "润色这句话："
        # 提取肯定情况
        part_after_affirm = query.split(marker_affirm, 1)[1]
        pos_section = part_after_affirm.split(tag, 1)[1]
        pos_text = pos_section.split("。", 1)[0].strip()
        # 提取否定情况
        part_after_neg = query.split(marker_neg, 1)[1]
        neg_section = part_after_neg.split(tag, 1)[1]
        neg_text = neg_section.split("。", 1)[0].strip()
        return user_said, pos_text, neg_text
    except Exception:
        raise ValueError("输入格式不匹配，请检查“用户说:…肯定…否定…”格式")

# ----------------------------
# 工具函数：调用 detect_client 判断肯定/否定
# ----------------------------
async def classify_affirmation(text: str, lang: str) -> str:
    prompt = classification_prompt_template.replace("{text}", text)
    resp = await asyncio.to_thread(
        detect_client.chat.completions.create,
        model="Qwen/Qwen3-0.6B",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    content = resp.choices[0].message.content
    print("content: ", content)
    # 用非贪婪正则，只抓第一个 {...}
    m = re.search(r"\{.*?\}", content, re.DOTALL)
    if not m:
        # 万一真没找到，默认否定
        return "否定"
    json_str = m.group(0)
    # 确保双引号格式合法
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # 如果意外出现单引号，简单替换再试
        fixed = json_str.replace("'", '"')
        data = json.loads(fixed)
    return data.get("sentiment", "否定")


# ----------------------------
# Stream LLM Helper
# ----------------------------
async def stream_llm(messages: List[Dict[str, str]], temperature: float, reply_accum: List[str]):
    stream = chat_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=messages,
        temperature=temperature,
        max_tokens=512,
        presence_penalty=0.5,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            reply_accum.append(token)
            yield token

# ----------------------------
# Main Endpoint：支持多轮 & SSE 流输出，增加诊断打印
# ----------------------------
@app.post("/chat/completions")
async def openai_compatible(request: Request):
    body = await request.json()
    if "messages" in body:
        try:
            raw = body["messages"][0]["content"]
            data = json.loads(raw)
            query = data["query"].strip()
            session_id = data["session_id"].strip()
            temperature = body.get("temperature", 0.7)
        except Exception:
            raise HTTPException(400, "Invalid messages content JSON")
    else:
        qr = QueryRequest(**body)
        query, session_id = qr.query.strip(), qr.session_id.strip()
        temperature = qr.temperature
    
    if session_id not in chat_history:
        chat_history[session_id] = deque(maxlen=MAX_TURNS * 2)

    try:
        user_said, pos_text, neg_text = parse_structured_input(query)
    except ValueError as e:
        raise HTTPException(400, str(e))
    print(f"[Session {session_id}] user_said: {user_said}")
    print(f"[Session {session_id}] pos_text: {pos_text}")
    print(f"[Session {session_id}] neg_text: {neg_text}")
    
    lang = detect_language(user_said)
    print(f"[Session {session_id}] lang: {lang}")
    
    
    sentiment = await classify_affirmation(user_said, lang)
    print(f"[Session {session_id}] classify_affirmation: {sentiment}")
    to_refine = pos_text if sentiment == "肯定" else neg_text
    print(f"[Session {session_id}] selected to_refine: {to_refine}")

    # 根据情感选择不同的润色模板
    if sentiment == "肯定":
        prompt_text = f"当前用户提问的语言是：{lang}，请一定要用{lang}回复，一定要以OK开头。\n\n" + polish_prompt_affirm.format(to_refine=to_refine, lang=lang)
    else:
        prompt_text = f"当前用户提问的语言是：{lang}，请一定要用{lang}回复。\n\n" + polish_prompt_neg.format(to_refine=to_refine, lang=lang)
    system_msg = {"role": "system", "content": prompt_text}

    history_msgs = list(chat_history[session_id])
    user_msg = {"role": "user", "content": to_refine}
    messages = [system_msg] + history_msgs + [user_msg]

    reply_accum: List[str] = []
    async def event_stream():
        async for token in stream_llm(messages, temperature, reply_accum):
            chunk = {"choices": [{"delta": {"content": token}, "finish_reason": ""}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    def final_res():
        full = "".join(reply_accum).strip()
        print(f"[Session {session_id}] polished reply: {full}")
        if full:
            chat_history[session_id].append({"role": "user", "content": user_said})
            chat_history[session_id].append({"role": "assistant", "content": full})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        background=BackgroundTask(final_res)
    )
    
    
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
        query = data["query"]
        session_id = data["session_id"]
        # 初始化或获取历史
        if session_id not in chat_history:
            chat_history[session_id] = deque(maxlen=MAX_TURNS * 2)
        # 拆解、分类、构建消息（同 HTTP 端点逻辑）
        user_said, pos_text, neg_text = parse_structured_input(query)
        lang = detect_language(user_said)
        print(f"[Session {session_id}] lang: {lang}")
        
        sentiment = await classify_affirmation(user_said, lang)
        
        to_refine = pos_text if sentiment == "肯定" else neg_text
        template = polish_prompt_affirm if sentiment == "肯定" else polish_prompt_neg
        system_content = (
            f"当前用户提问的语言是：{lang}，请始终用{lang}回复。"
            + template.format(to_refine=to_refine, lang=lang)
        )
        msgs = [{"role": "system", "content": system_content}] + list(chat_history[session_id]) + [{"role":"user","content":to_refine}]
        # 实时流式发送
        acc = []
        for chunk in chat_client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct", messages=msgs, stream=True
        ):
            token = chunk.choices[0].delta.content or ""
            if token:
                acc.append(token)
                await ws.send_text(token)
        await ws.send_text("[DONE]")
        # 保存会话历史
        full = "".join(acc).strip()
        chat_history[session_id].append({"role":"user","content":user_said})
        chat_history[session_id].append({"role":"assistant","content":full})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(f"ERROR: {e}")
        await ws.close()