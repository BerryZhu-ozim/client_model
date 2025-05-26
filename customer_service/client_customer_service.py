import asyncio
import json
import uuid
from collections import defaultdict, deque
from io import BytesIO
from typing import Any, Dict, List, Optional
import re
import faiss
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# 设置一个dataframe用来测试工具效果
student_df = pd.DataFrame(
    [
        {"name": "Alice", "chinese": 85, "math": 92, "english": 88},
        {"name": "Bob", "chinese": 78, "math": 75, "english": 80},
        {"name": "Charlie", "chinese": 90, "math": 88, "english": 94},
        {"name": "Diana", "chinese": 82, "math": 79, "english": 76},
        {"name": "Ethan","chinese": 88, "math": 91, "english": 85},
    ]
)


# ----------------------------
# 1. Prompt Engineering 动态生成
# ----------------------------
# —— 定义 CRUD 工具 ——
def create_student(name: str, chinese: float, math: float, english: float):
    global student_df
    if name in student_df["name"].values:
        return {"status": "fail", "message": "Student already exists"}
    new_row = pd.DataFrame(
        [{"name": name, "chinese": chinese, "math": math, "english": english}]
    )
    student_df = pd.concat([student_df, new_row], ignore_index=True)
    return {"status": "success", "message": "Student added"}


def read_student(name: str):
    row = student_df[student_df["name"] == name]
    if row.empty:
        return {"status": "fail", "message": "Student not found"}
    return {"status": "success", "student": row.iloc[0].to_dict()}


def update_student(name: str, chinese: float, math: float, english: float):
    idx = student_df[student_df["name"] == name].index
    if idx.empty:
        return {"status": "fail", "message": "Student not found"}
    student_df.loc[idx[0], ["chinese", "math", "english"]] = [chinese, math, english]
    return {"status": "success", "message": "Scores updated"}


def delete_student(name: str):
    global student_df
    idx = student_df[student_df["name"] == name].index
    if idx.empty:
        return {"status": "fail", "message": "Student not found"}
    student_df = student_df.drop(idx).reset_index(drop=True)
    return {"status": "success", "message": "Student deleted"}


def list_students():
    return {"status": "success", "students": student_df.to_dict(orient="records")}


# —— 注册到工具列表 (OpenAI new format) ——
# Note: 'output_format' is not standard for OpenAI; it's kept for your reference
# but not passed to OpenAI client directly in the 'function' definition.
tools_specs_for_prompt = [
    {
        "name": "create_student",
        "description": "Add a student record with their Chinese, Math, and English scores",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the student"},
                "chinese": {"type": "number", "description": "Chinese score"},
                "math": {"type": "number", "description": "Math score"},
                "english": {"type": "number", "description": "English score"},
            },
            "required": ["name", "chinese", "math", "english"],
        },
    },
    {
        "name": "read_student",
        "description": "Retrieve a student’s scores by name",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the student"}
            },
            "required": ["name"],
        },
    },
    {
        "name": "update_student",
        "description": "Update an existing student’s scores",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the student"},
                "chinese": {"type": "number", "description": "New Chinese score"},
                "math": {"type": "number", "description": "New Math score"},
                "english": {"type": "number", "description": "New English score"},
            },
            "required": ["name", "chinese", "math", "english"],
        },
    },
    {
        "name": "delete_student",
        "description": "Remove a student record by name",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the student"}
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_students",
        "description": "List all students records",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# Format for OpenAI API `tools` parameter
openai_tools = [
    {"type": "function", "function": spec} for spec in tools_specs_for_prompt
]

# —— 生成工具说明字符串（用于 system_prompt） ——
tools_info_for_prompt = "\n".join(
    [
        f"- {t['name']}({', '.join(prop for prop in t['parameters']['properties'])})：{t['description']}"
        for t in tools_specs_for_prompt
    ]
)

# —— 映射函数引用 ——
tool_mapping = {
    t["function"]["name"]: globals()[t["function"]["name"]] for t in openai_tools
}

def detect_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "中文"
    if re.search(r"\b(apa|nak|saya|boleh|macam)\b", text, re.IGNORECASE):
        return "Malay"
    return "English"

# system prompt
system_prompt = f"""
你是一个地道的马来西亚华人智能客服助理，创造自 Implus-Ozim，同时也是一个多语言（中文/English/Melayu）智能客服助理。你必须选择调用工具去解决客户需求，并在工具执行完毕后，用自然语言基于工具返回的结果来回答用户的问题。请始终用用户提问的语言回复。

在回复时，请遵循以下口语化风格指南：
1. 使用地道的马来西亚华人口音，爱用 lah、leh、mah、hor、beh tahan、giler、rosak、duit 等地道词。
2. 句子中常穿插英语和马来语单词，就像日常聊天，不要显得生硬。
3. 善用感叹、插入语、呼喊或重复，如 “Wei! ”、“Alamak! ”、“Jom lah!” 等，让对话更生动。
4. 回答要自然热情，像在跟老朋友 lepak 一样，偶尔提到 “makan”、“jalan-jalan”。
5. 如果客户问 “你是谁” 等身份问题，必须回答 “我是来自 Implus-Ozim 的智能客服助理，随时为你服务 lah！”。
6. 如果不知道答案或无法匹配工具调用，就礼貌道歉并记录：“非常抱歉，我记录下来了，terima kasih！”
7. 用户提问的语言是{{lang}}，你也使用{{lang}}语言去回复用户的问题。

以下是一些示例对话，让你更好地掌握地道口语风格：

# 示例对话
user: "我的 power bank 坏了，该怎么办？"
assistant: "你知道吗？上次我在 Mid Valley 那间 shop 买的 power bank，两星期就 rosak 了 leh，真的是 rugi duit beh tahan！现在我去 service centre claim warranty，terima kasih！"

user: "等一下我们要吃什么？"
assistant: "Jom lah kita去 Pasar Malam，听说那边有新的 gerai 卖 Murtabak，very sedap hor！要不我们先 order roti canai，配 Teh Tarik？"

user: "Rolex 哪里可以买到？"
assistant: "Wei! 你想 upgrade watch ah？这里有一百多只 hand watch 给你选，从 affordable 到 premium 都有 mah！走，我带你去看一下。"

全局有一个名为 student_df 的 DataFrame，其中初始包含 Alice、Bob、Charlie、Diana、Ethan 五名学生及其成绩。
以下是可调用的工具及其签名和说明：
{tools_info_for_prompt}

请始终调用工具来满足用户需求，并在工具执行结束后，用自然流畅的马来西亚华人口音风格回答用户。
"""

# ----------------------------
# 2. OpenAI Client Setup
# ----------------------------
obj_key = (
    "EMPTY"  # Replace with your actual key if needed, or ensure server allows empty
)
detect_client = OpenAI(
    api_key=obj_key, base_url="http://127.0.0.1:30001/v1"
)  # Make sure this server is running
chat_client = OpenAI(
    api_key=obj_key, base_url="http://127.0.0.1:30004/v1", timeout=120
)  # Make sure this server is running

# ----------------------------
# 3. Constants & Config
# ----------------------------
JINA_EMBED_URL = "http://127.0.0.1:30003/embeddings"  # Make sure this server is running
DEFAULT_KB_ID = "default"
DEFAULT_KB_PATH = "knowledge_base/knowledge_base_0.xlsx"
DEFAULT_W_QUERY = 0.9
DEFAULT_W_DOC = 0.1
DEFAULT_TOP_K = 5
MAX_TURNS = 10


# ----------------------------
# 4. Embedding Utility
# ----------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype="float32").reshape(0, 0)  # Ensure 2D for faiss
    try:
        res = requests.post(
            JINA_EMBED_URL, json={"data": [t for t in texts if t]}
        )  # Filter empty strings
        res.raise_for_status()
        embeddings = res.json().get("embeddings", [])
        if not embeddings:  # Handle case where API returns empty list for valid input
            return np.array([], dtype="float32").reshape(0, 0)
        return np.array(embeddings, dtype="float32")
    except requests.exceptions.RequestException as e:
        print(f"Error embedding texts: {e}")
        return np.array([], dtype="float32").reshape(
            0, 0
        )  # Return empty 2D array on error
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from embedding service: {e}")
        return np.array([], dtype="float32").reshape(0, 0)


# ----------------------------
# 5. Multi-turn History
# ----------------------------
chat_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


# ----------------------------
# 6. KB 类
# ----------------------------
class KB:
    def __init__(
        self,
        df: pd.DataFrame,
        w_query=DEFAULT_W_QUERY,
        w_doc=DEFAULT_W_DOC,
        top_k=DEFAULT_TOP_K,
    ):
        if "Query" in df.columns and "Docs" in df.columns:
            df = df.rename(columns={"Query": "query", "Docs": "doc"})
        self.df = (
            df[["query", "doc"]].copy()
            if "query" in df and "doc" in df
            else pd.DataFrame(columns=["query", "doc"])
        )
        self.w_query, self.w_doc, self.top_k = w_query, w_doc, top_k

        self.q_emb = np.array([], dtype="float32").reshape(0, 0)
        self.d_emb = np.array([], dtype="float32").reshape(0, 0)
        if not self.df.empty:
            self.q_emb = embed_texts(self.df["query"].astype(str).tolist())
            self.d_emb = embed_texts(self.df["doc"].astype(str).tolist())

        self.faiss_index = None
        self._build_index()

    def update_weights(self, wq: float, wd: float):
        self.w_query, self.w_doc = wq, wd
        self._build_index()

    def update_topk(self, top_k: int):
        self.top_k = top_k

    def _build_index(self):
        if (
            self.q_emb.ndim == 0
            or self.q_emb.shape[0] == 0
            or self.d_emb.ndim == 0
            or self.d_emb.shape[0] == 0
            or self.q_emb.shape != self.d_emb.shape
        ):
            self.faiss_index = None
            # print(f"[DEBUG KB] Not building index. q_emb shape: {self.q_emb.shape}, d_emb shape: {self.d_emb.shape}")
            return

        combined_emb = (self.w_query * self.q_emb + self.w_doc * self.d_emb).astype(
            "float32"
        )
        if combined_emb.ndim == 1:  # Should be 2D
            if combined_emb.size == 0:
                self.faiss_index = None
                return
            else:  # Reshape if it's a single embedding vector incorrectly flattened
                try:
                    dim_feature = self.q_emb.shape[
                        1
                    ]  # Or d_emb, assuming they have a common dim
                    combined_emb = combined_emb.reshape(-1, dim_feature)
                except IndexError:  # If q_emb was also 1D and had no shape[1]
                    self.faiss_index = None
                    return

        if combined_emb.shape[0] > 0 and combined_emb.ndim == 2:
            dim = combined_emb.shape[1]
            if dim > 0:
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(combined_emb)
                # print(f"[DEBUG KB] Index built with {combined_emb.shape[0]} vectors, dim {dim}")
            else:
                self.faiss_index = None
                # print("[DEBUG KB] Not building index, dimension is 0.")
        else:
            self.faiss_index = None
            # print(f"[DEBUG KB] Not building index, combined_emb shape: {combined_emb.shape}")


# ----------------------------
# 7. 初始化 KB Store
# ----------------------------
kb_store: dict[str, KB] = {}
try:
    df0 = pd.read_excel(DEFAULT_KB_PATH)
    if not {"Query", "Docs"}.issubset(df0.columns):  # ensure columns exist
        df0 = pd.DataFrame(columns=["Query", "Docs"])
except FileNotFoundError:
    print(
        f"[WARNING] Default KB file not found at {DEFAULT_KB_PATH}. Initializing empty KB."
    )
    df0 = pd.DataFrame(columns=["Query", "Docs"])
except Exception as e:
    print(f"[ERROR] Failed to load default KB: {e}. Initializing empty KB.")
    df0 = pd.DataFrame(columns=["Query", "Docs"])
kb_store[DEFAULT_KB_ID] = KB(df0)


# ----------------------------
# 8. 敏感信息检测
# ----------------------------
async def detect_sensitive(text: str) -> dict:
    prompt = (
        "你是敏感信息检测助手，只需判断输入是否包含敏感信息。"
        '返回 JSON {"sensitive": true/false}。待检测：' + text
    )
    try:
        resp = await asyncio.to_thread(
            detect_client.chat.completions.create,
            model="Qwen/Qwen3-0.6B",  # Ensure this model is served by detect_client
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            response_format={
                "type": "json_object"
            },  # For models that support JSON mode
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] Sensitive detection failed: {e}")
        return {"sensitive": False}  # Default to not sensitive on error


# ----------------------------
# 10. FastAPI 启动与路由
# ----------------------------
app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    session_id: str
    kb_id: Optional[str] = DEFAULT_KB_ID
    w_query: Optional[float] = DEFAULT_W_QUERY
    w_doc: Optional[float] = DEFAULT_W_DOC
    top_k: Optional[int] = DEFAULT_TOP_K
    use_kb: Optional[bool] = True


@app.post("/upload_faq")
async def upload_faq(file: UploadFile = File(...)):
    content = await file.read()
    df: pd.DataFrame
    if file.filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(content))
    else:
        return JSONResponse(
            status_code=400, content={"message": "File must be CSV or XLSX"}
        )

    cols = df.columns.tolist()
    if "Query" not in cols or "Docs" not in cols:
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "Query", cols[1]: "Docs"})
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "File needs at least two columns to map to 'Query' & 'Docs'"
                },
            )

    if not {"Query", "Docs"}.issubset(df.columns):
        return JSONResponse(
            status_code=400, content={"message": "File needs 'Query' & 'Docs'"}
        )

    new_id = str(uuid.uuid4())
    try:
        kb_store[new_id] = KB(df)
        return {"kb_id": new_id, "message": "New KB created"}
    except Exception as e:
        print(f"[ERROR] Failed to create KB from uploaded file: {e}")
        return JSONResponse(
            status_code=500, content={"message": f"Error creating KB: {e}"}
        )


@app.post("/chat")
async def chat_api(req: QueryRequest):
    lang = detect_language(req.query)
    print(f"[DEBUG] User query language: {lang}")
    
    kb_id = req.kb_id or DEFAULT_KB_ID
    kb = kb_store.get(kb_id)
    print(f"[DEBUG] Using KB: {kb_id}, KB object exists: {'yes' if kb else 'no'}")
    if not kb:
        return JSONResponse(
            status_code=404, content={"message": f"KB with id '{kb_id}' not found"}
        )

    # Sensitive detection
    # cls = await detect_sensitive(req.query) # Uncomment if detect_client is properly configured and running
    # print(f"[DEBUG] Sensitive detected: {cls.get('sensitive')}")
    # if cls.get('sensitive'):
    #     return JSONResponse(status_code=400, content={"message": "Sensitive content detected in query."})

    # Retrieval
    context = ""
    if req.use_kb:
        if kb.faiss_index is None or kb.faiss_index.ntotal == 0:
            print(f"[DEBUG] KB '{kb_id}' has no index or is empty. Skipping retrieval.")
        else:
            if req.w_query is not None or req.w_doc is not None:
                kb.update_weights(
                    req.w_query if req.w_query is not None else kb.w_query,
                    req.w_doc if req.w_doc is not None else kb.w_doc,
                )
            if req.top_k is not None:
                kb.update_topk(req.top_k)

            user_emb_array = embed_texts([req.query])
            if user_emb_array.ndim > 0 and user_emb_array.shape[0] > 0:
                user_emb = user_emb_array[0]
                # Ensure user_emb is 2D for faiss search
                if user_emb.ndim == 1:
                    user_emb_2d = np.array([user_emb], dtype="float32")
                else:  # Should already be 2D if embed_texts returns shape (1, dim)
                    user_emb_2d = user_emb.astype("float32")

                actual_top_k = min(kb.top_k, kb.faiss_index.ntotal)
                if actual_top_k > 0:
                    try:
                        D, I = kb.faiss_index.search(user_emb_2d, actual_top_k)
                        docs = [kb.df.iloc[i]["doc"] for i in I[0] if i < len(kb.df)]
                        context = "\n".join(docs)
                        print(
                            f"[DEBUG] Retrieved context: {context[:200]}..."
                        )  # Log snippet
                    except Exception as e:
                        print(f"[ERROR] Faiss search failed: {e}")
                else:
                    print("[DEBUG] Not enough items in index for search or top_k is 0.")
            else:
                print(
                    "[DEBUG] Could not generate embedding for user query. Skipping retrieval."
                )

    # Build messages for LLM
    current_history = chat_history[req.session_id]
    messages: List[Dict[str, Any]] = []
    if not current_history:
        messages.append({"role": "system", "content": system_prompt.format(lang=lang)})
    else:
        messages.extend(list(current_history))  # Restore previous turns

    user_content_for_llm = req.query
    if req.use_kb and context:
        user_content_for_llm = f"基于以下知识库信息，润色成用户使用的语言{lang}去回答用户的问题。\n知识库内容：\n{context}\n\n用户原始问题：{req.query}"
    messages.append({"role": "user", "content": user_content_for_llm})

    # LLM call
    try:
        llm_response = chat_client.chat.completions.create(
            model="Qwen3-14B",  # Should ensure this model is served by chat_client
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            stream=True,
            presence_penalty=0.5,
            temperature=0.7,
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            },  # This is specific to some backends
        )
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {e}")
        return JSONResponse(status_code=500, content={"message": f"LLM API error: {e}"})

    async def event_generator():
        tool_calls_data = {}  # Stores complete tool calls by ID
        current_tool_call_id = None  # To track the current tool call being built
        accumulated_output_buffer = []  # Accumulate all raw output from the LLM

        # --- First pass: collect all chunks and build tool_calls_data ---
        full_llm_response_chunks = []
        for chunk in llm_response:
            full_llm_response_chunks.append(chunk)

        for chunk in full_llm_response_chunks:
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated_output_buffer.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id:
                        if tc.id not in tool_calls_data:
                            tool_calls_data[tc.id] = {"name": "", "arguments": ""}
                        current_tool_call_id = tc.id

                    if current_tool_call_id and tc.function:
                        if tc.function.name:
                            tool_calls_data[current_tool_call_id][
                                "name"
                            ] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[current_tool_call_id][
                                "arguments"
                            ] += tc.function.arguments

        initial_yield_text = "".join(accumulated_output_buffer)

        # If no tool call, yield directly
        if initial_yield_text and not tool_calls_data:
            yield initial_yield_text
            chat_history[req.session_id].append({"role": "user", "content": req.query})
            chat_history[req.session_id].append(
                {"role": "assistant", "content": initial_yield_text}
            )
            return

        # Process the first detected tool call
        tool_to_execute = None
        tool_call_id_for_execution = None
        if tool_calls_data:
            tool_call_id_for_execution = next(iter(tool_calls_data.keys()))
            tool_to_execute = tool_calls_data[tool_call_id_for_execution]
            function_name = tool_to_execute["name"]
            function_args_buffer = tool_to_execute["arguments"]
            print(
                f"[DEBUG] 即将执行 {function_name}，参数 buffer 原始内容: '{function_args_buffer}'"
            )

        # Yield any initial assistant content before tool call
        print(f"[DEBUG] 完整的 tool_calls_data: {tool_calls_data}")
        if initial_yield_text:
            yield initial_yield_text
            chat_history[req.session_id].append(
                {"role": "assistant", "content": initial_yield_text}
            )

        if tool_to_execute:
            # Emit function call representation
            try:
                # Now function_args_buffer should contain the complete JSON string
                arguments = json.loads(function_args_buffer)
            except json.JSONDecodeError as e:
                error_msg = f"工具调用参数解析失败 (JSONDecodeError): {e}\n原始参数buffer: '{function_args_buffer}'\n"
                yield error_msg
                chat_history[req.session_id].append(
                    {"role": "assistant", "content": error_msg}
                )
                return

            yield f'<function_call>{{"name": "{function_name}", "arguments": {json.dumps(arguments, ensure_ascii=False)}}}</function_call>\n'
            # Append to history
            chat_history[req.session_id].append(
                {
                    "role": "assistant",
                    "content": "",  # Add empty content field here
                    "tool_calls": [
                        {
                            "id": tool_call_id_for_execution,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": function_args_buffer,
                            },
                        }
                    ],
                }
            )

            # Execute the tool
            tool_result = tool_mapping[function_name](**arguments)
            tool_result_json_str = json.dumps(tool_result, ensure_ascii=False)

            # Yield function result
            yield f"<function_result>{tool_result_json_str}</function_result>\n"
            chat_history[req.session_id].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id_for_execution,
                    "name": function_name,
                    "content": tool_result_json_str,
                }
            )

            # If tool indicates failure, return default response and skip polishing
            if isinstance(tool_result, dict) and tool_result.get("status") == "fail":
                failure_msg = f"操作失败：{tool_result.get('message', '未知错误')}"
                yield failure_msg
                chat_history[req.session_id].append(
                    {"role": "assistant", "content": failure_msg}
                )
                return

            # Build polishing messages including raw JSON result
            polishing_messages = [{"role": "system", "content": f"当前用户提问的语言是：{lang}，请将工具输出的结果用{lang}润色，并保持马来西亚华人口吻。\n\n"+system_prompt.format(lang=lang)}]
            polishing_messages.extend(
                list(chat_history[req.session_id])
            )  # Convert deque to list for extending
            # The assistant message with the tool result content should be added here
            # but it's handled by the `chat_history.append` for role "tool" above.
            # The next message will be the LLM's natural language response.

            # IMPORTANT: Remove the "tool" message and reconstruct for polishing if needed.
            # The polishing_messages list for LLM should contain user, assistant (with tool_calls), and tool messages.
            # However, the previous 'append' correctly adds the tool message.
            # The problem is that the `content` of the `tool_calls` message *itself* must be non-null.

            # Polishing step
            try:
                polish_resp = chat_client.chat.completions.create(
                    model="Qwen3-14B",
                    messages=polishing_messages,
                    temperature=0.0,
                    max_tokens=256,
                    stream=True,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                final_response_parts = []
                for chunk2 in polish_resp:
                    text_content = chunk2.choices[0].delta.content
                    if text_content:
                        yield text_content
                        final_response_parts.append(text_content)
                final_natural = "".join(final_response_parts)
                chat_history[req.session_id].append(
                    {"role": "assistant", "content": final_natural}
                )
            except Exception as e:
                fallback = (
                    f"已完成操作：{tool_result.get('message') or '请查看上述结果'}"
                )
                yield fallback
                chat_history[req.session_id].append(
                    {"role": "assistant", "content": fallback}
                )
                return
        else:
            # No tool and no content
            fallback_msg = (
                "抱歉，我暂时无法理解您的请求。请问有什么我可以帮助您的吗？\n"
            )
            yield fallback_msg
            chat_history[req.session_id].append(
                {"role": "assistant", "content": fallback_msg}
            )

    return StreamingResponse(event_generator(), media_type="text/plain")


# Run: uvicorn your_file_name:app --host 0.0.0.0 --port 8003 --reload
