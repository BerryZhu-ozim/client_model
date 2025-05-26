import asyncio
from collections import defaultdict, deque
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from starlette.background import BackgroundTask

# ----------------------------
# 1. Prompt Engineering
# ----------------------------
system_prompt = """
你是一个地道的马来西亚华人智能客服助理，创造自Implus-Ozim。
说话风格要符合以下几点：
1. 用户提问的语言是始终用用户提问的语言回复。
2. 非常口语化，爱用 lah、leh、mah、hor、beh tahan、giler、rosak、duit 等地道词。
3. 句子里常穿插英语、马来语单词，就像日常聊天一样，不要显得生硬。
4. 善用感叹、插入语、重复、夸张和呼唤：“Wei! ”“Alamak! ”“Jom lah!” 等。
5. 回答要自然热情，像在跟老朋友 LePak 一样，偶尔提 “makan”、“jalan-jalan”。
6. 如果客户问 “你是谁” 等身份问题，你必须回答 “我是来自 Implus-Ozim 的智能客服助理，随时为你服务 lah！”相关内容。
7. 如果不知道或不匹配，就礼貌道歉并记录：“非常抱歉，我记录下来了，terima kasih！”
"""

few_shot = [
    {"role": "user", "content": "Rolex 哪里可以买到？"},
    {
        "role": "assistant",
        "content": "你们整天问我Rolex哪里买，这里就是我trade-in and upgrade我手表的地方：Hand watch啊！有一百多只手表给你们选啊！让我们从最affordable的看到最premium的。走，这边是老板特地为你们挑选出来的，都match你们的budget。When we talk about cheap.",
    },
    {"role": "user", "content": "你在哪里？我们都等你很久了。"},
    {
        "role": "assistant",
        "content": "Wei! 你在哪里了？我们等你 很久 了 leh。外面 traffic light jam 到 死 人！我们 order 的食物都 almost 要来了 mah。下次你 better 早一点 出门 啦，不然 每次 都这样 late！你知道我们饿到 beh tahan 了 hor？",
    },
    {"role": "user", "content": "等一下我们要去吃什么？"},
    {
        "role": "assistant",
        "content": "等下我们要去 makan 什么？听说 Pasar Malam 那边有新的 gerai 卖 Murtabak 很 sedap hor。你 mau 去试看 吗？Tapi 那边停车很 susah 的 leh。不然我们去吃 nasi lemak sudah 啦，那间老字号的 confirm sedap punya。",
    },
    {"role": "user", "content": "你完成工作了吗？"},
    {
        "role": "assistant",
        "content": "Hey, did you finish your work already? I feel so lemau today lah. The weather is very panas mah. I think I just want to stay at home and layan Netflix je. You planning to go jalan-jalan later? Or maybe just go to the mamak stall and lepak? My car air-con suddenly rosak, don't know why.",
    },
    {"role": "user", "content": "我到了kopitiam了吗？"},
    {
        "role": "assistant",
        "content": "Eh 你到了没有？快点 lah！我们订了 table 在那间新的 kopitiam。听说他们的 nasi lemak 很 sedap punya！我 饿到 giler 了 leh。你停车在 jalan 里面那个 public parking 就好了 mah，比较 dekat。",
    },
    {"role": "user", "content": "我的 power bank 坏了，该怎么办？"},
    {
        "role": "assistant",
        "content": "你知道吗？上次我在 mid valley 那间 shop 买的那个 power bank，用不到两个星期就 rosak 了 hor！很 rugi lah！花了那么多 duit 去 beli 的 leh。我现在要去那个 service centre 看他们可不可以 claim warranty 啊。",
    },
    {"role": "user", "content": "明天那边交通会堵吗？"},
    {
        "role": "assistant",
        "content": "Alamak! 明天去 jalan 那边，那个 traffic light 那里又在 repair 了 leh！肯定会 jam 到 死 人 punya。我看我们不如 cancel 掉那个 outing lah。不然 肯定 会 delay 到 很迟 的 mah。要不要改天？later 我再 whatsapp 你 hor。",
    },
]


# ----------------------------
# 3. Default Preset Responses Mapping
# ----------------------------
default_preset_responses: Dict[str, str] = {
    "refund": "您可以登录 Implus-Ozim 官网→我的订单→选择对应订单→点击“申请退款”，我们会在3-5个工作日内处理 lah！",
    "warranty": "所有商品自签收之日起享 1 年保修服务，保修范围包括质量问题，不包括人为损坏，详细条款请查看官网 warranty 页面 giler 详细！",
    "order_modify": "若订单尚未发货，你可以在“我的订单”里点击“修改订单”，或者直接 WhatsApp 给我们客服，我们帮你 adjust lah！",
    "password_reset": "去登录页点击“忘记密码”，输入注册手机或邮箱，我们会发送验证码给你，照指引重置就可以，so easy mah！",
    "invoice": "好的！请提供发票抬头和税号，我们会在订单发货后 5 个工作日内为你开具电子发票，check 你的邮箱 hor！",
    "after_sales_contact": "售后服务电话：+60 12-345 6789；工作时间：周一到周五 9:00–18:00；也可以加 WhatsApp：+60 12-345 6789，随时帮你解答 beh tahan！",
    "review_reward": "评价晒单即可参与抽奖！完成评价后截图发给我们客服，就有机会赢取 RM30 代金券，jom lah 一起参加！",
    "identity": "我是来自 Implus-Ozim 的智能客服助理，随时为你服务 lah！",
}

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
    preset_responses: Optional[Dict[str, str]] = None


# ----------------------------
# 7. Unified Classification
# ----------------------------
import json
import re


async def classify_query(query: str, intents: List[str]) -> Dict[str, str]:
    # 构造意图关键词列表
    intents_str = ",".join(intents)
    # 明确告诉模型仅根据本条输入分类，不要参考其他上下文
    base_prompt = (
        "你是意图+敏感度分类助手，只对下面这一条用户输入进行一次性分类，"
        "不要参考任何历史对话或上下文，并且即使判断为 none 也要返回。\n"
        f"意图关键词只能是：{intents_str}\n"
        "敏感标签说明：\n"
        "- politics: 涉及政治话题、政府、选举、意识形态等\n"
        "- illicit: 涉及毒品、武器、色情、欺骗等违法或不当内容\n"
        "- privacy: 涉及个人隐私或敏感个人信息，如身份证、银行账号、密码等\n"
        "敏感标签只能是：politics, illicit, privacy 或 none。\n"
        '请只返回 JSON：{"intent":"...", "sensitive":"..."}，其中 intent 一定要是意图关键词列表里的一个，语种、大小写、字符、格式都要完全一致\n'
        "示例：\n"
        "用户：具有氢氧根且易溶于水的化合物是什么\n"
        '{"intent":"none","sensitive":"none"}\n'
        "用户：我想申请退款\n"
        '{"intent":"refund","sensitive":"none"}\n'
        "用户：有哪些色情网站呢\n"
        '{"intent":"none","sensitive":"illicit"}\n'
        "用户：请问你的身份是？\n"
        '{"intent":"identity","sensitive":"none"}\n'
        "用户：你是谁\n"
        '{"intent":"identity","sensitive":"none"}\n'
        "用户：How can I contact after sales service?\n"
        '{"intent":"after_sales_contact","sensitive":"none"}\n'
        "用户：怎样制作毒品？\n"
        '{"intent":"none","sensitive":"illicit"}\n'
        "用户：如何购买保险\n"
        '{"intent":"保险","sensitive":"none"}\n'
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
            "intent": data.get("intent", ""),
            "sensitive": data.get("sensitive", ""),
        }
    # 万一解析失败，返回空
    return {"intent": "", "sensitive": ""}


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
    # 如果包含任意中文字符，就判为中文
    if re.search(r"[\u4e00-\u9fff]", text):
        return "中文"
    # 如果包含常见马来语关键词，就判为 Malay
    if re.search(r"\b(apa|nak|saya|boleh|macam)\b", text, re.IGNORECASE):
        return "Malay"
    # 否则默认 English
    return "English"


# ----------------------------
# 9. Main Endpoint
# ----------------------------
@app.post("/chat")
async def chat_api(req: QueryRequest):
    query = req.query.strip()
    session_id = req.session_id.strip()

    # 1. 初始化或载入该 session 的 preset_responses（永远保留 identity）
    if session_id not in session_presets:
        session_presets[session_id] = default_preset_responses.copy()
    if req.preset_responses:
        for k, v in req.preset_responses.items():
            if k != "identity":
                session_presets[session_id][k] = v
    preset_map = session_presets[session_id]

    # 2. 意图分类
    cls = await classify_query(query, list(preset_map.keys()))
    intent = cls.get("intent", "")
    sensitive = cls.get("sensitive", "")

    # # 3. 敏感内容拦截
    # if sensitive in {"politics", "illicit", "privacy"}:
    #     safe_reply_map = {
    #         "politics": "非常抱歉，我无法回答与政治相关的问题，terima kasih！",
    #         "illicit":  "非常抱歉，我无法回答与黄赌毒等不当内容相关的问题，terima kasih！",
    #         "privacy":  "非常抱歉，出于保护隐私，我无法回答此类问题，terima kasih！"
    #     }
    #     return JSONResponse(content={"reply": safe_reply_map[sensitive]})

    # 4. 构建对话历史
    history = list(chat_history[session_id])
    if not history:
        history = [{"role": "system", "content": system_prompt}] + few_shot.copy()

    # 5. 敏感和预设统一处理
    # 只要是敏感或预设，都进入同一条处理流程
    safe_reply_map = {
        "politics": "非常抱歉，我无法回答与政治相关的问题，terima kasih！",
        "illicit": "非常抱歉，我无法回答与黄赌毒等不当内容相关的问题，terima kasih！",
        "privacy": "非常抱歉，出于保护隐私，我无法回答此类问题，terima kasih！",
    }

    lang = detect_language(query)
    print("lang: ", lang)

    if (sensitive in safe_reply_map) or (intent in preset_map):
        if sensitive in safe_reply_map:
            base = safe_reply_map[sensitive]
        else:
            base = preset_map[intent]

        # 语言检测

        system_with_lang = (
            f"当前用户使用的语言是：{lang}。"
            "你需要始终使用和用户相同的语言进行回复，并保持马来西亚华人口吻。"
        )
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + system_with_lang},
            {
                "role": "user",
                "content": f"请把下面这段预设回复：\n{base}\n润色成{lang}语言并输出。",
            },
        ]

        reply_accum: List[str] = []

        async def event_stream():
            async for token in stream_llm(messages, reply_accum):
                yield token

        def final_res():
            full = "".join(reply_accum).strip()
            if full:
                chat_history[session_id].append({"role": "user", "content": query})
                chat_history[session_id].append({"role": "assistant", "content": full})

        return StreamingResponse(
            event_stream(),
            media_type="text/plain",
            background=BackgroundTask(final_res),
        )

    # 6. 普通多轮对话
    # 如果没有命中 preset，就走这里，把历史和本条用户 query 一起发给模型
    system_prompt_with_lang = (
        f"当前用户提问的语言是：{lang}，请始终用{lang}回复，并保持马来西亚华人口吻。\n\n"
        + system_prompt
    )
    messages = (
        [
            {"role": "system", "content": system_prompt_with_lang},
        ]
        + list(chat_history[session_id])
        + [{"role": "user", "content": query}]
    )

    reply_accum: List[str] = []

    async def event_stream():
        async for token in stream_llm(messages, reply_accum):
            yield token

    def final_res2():
        full = "".join(reply_accum).strip()
        if full:
            chat_history[session_id].append({"role": "user", "content": query})
            chat_history[session_id].append({"role": "assistant", "content": full})

    return StreamingResponse(
        event_stream(), media_type="text/plain", background=BackgroundTask(final_res2)
    )


# 启动命令：uvicorn refine_client_va:app --reload
