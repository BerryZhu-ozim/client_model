# import requests
# import socket
# import json
# import sys

# # 1. 配置
# def get_local_ip():
#     return socket.gethostbyname(socket.gethostname())

# IP = get_local_ip()
# CHAT_URL = f"http://{IP}:8003"
# EMBED_URL = f"http://{IP}:30003"
# CHAT_API = CHAT_URL + "/chat"
# UPLOAD_URL = CHAT_URL + "/upload_faq"
# EMBED_API = EMBED_URL + "/embeddings"

# # 2. 辅助函数
# def check_service(name, docs_url):
#     try:
#         r = requests.get(docs_url)
#         print(f"[CHECK] {name} → {r.status_code}")
#     except Exception as e:
#         print(f"[ERROR] 无法访问 {name}: {e}")
#         sys.exit(1)

# def print_resp(r):
#     print("  Status:", r.status_code)
#     # print("  Body  :", r.text.strip())

# def stream_chat(payload):
#     """
#     对 /chat 调用启用 stream=True，并实时打印 SSE 风格的 data 消息
#     """
#     print(f"\n>> Payload: {json.dumps(payload, ensure_ascii=False)}")
#     with requests.post(CHAT_API, json=payload, stream=True) as resp:
#         print("Status:", resp.status_code)
#         # iter_lines 遇到 \n\n 才返回一段内容
#         for line in resp.iter_lines(decode_unicode=True):
#             if not line:
#                 continue
#             # 假设服务器直接返回内容（text/plain），直接打印
#             # 如果使用 SSE，需要去掉开头的 "data: "
#             if line.startswith("data:"):
#                 content = line[len("data:"):].lstrip()
#             else:
#                 content = line
#             # 不换行继续打印，flush 确保即时输出
#             print(content, end="", flush=True)
#         print("\n--- Stream End ---")

# # 3. Test embedding service（同步）
# def test_embedding():
#     print("\n=== Test Embedding Service ===")
#     payload = {"data": ["Hello", "测试"]}
#     r = requests.post(EMBED_API, json=payload)
#     print_resp(r)

# # 4. Test default chat (with KB, streaming)
# def test_default_chat():
#     print("\n=== Test Default KB Chat (use_kb=True) ===")
#     payload = {
#         "query": "你好，请介绍一下你自己。",
#         "session_id": "test_default",
#         "use_kb": True
#     }
#     stream_chat(payload)

# # 5. Test chat without KB (streaming)
# def test_chat_without_kb():
#     print("\n=== Test Chat Without KB (use_kb=False) ===")
#     payload = {
#         "query": "今天天气怎么样？",
#         "session_id": "test_nokb",
#         "use_kb": False
#     }
#     stream_chat(payload)

# # 6. Upload a new KB（同步）
# def test_upload_kb():
#     print("\n=== Test Upload New KB ===")
#     csv_data = "Query,Docs\nFoo,Response for Foo\nBar,Response for Bar\n"
#     files = {"file": ("kb.csv", csv_data)}
#     r = requests.post(UPLOAD_URL, files=files)
#     print_resp(r)
#     try:
#         return r.json().get("kb_id")
#     except:
#         return None

# # 7. Test weight & top_k via /chat optional params (streaming)
# def test_weights_and_topk(kb_id):
#     print("\n=== Test /chat with weight override (use_kb=True) ===")
#     payload = {
#         "query": "Foo",
#         "session_id": "s1",
#         "kb_id": kb_id,
#         "w_query": 0.2,
#         "w_doc":   0.8,
#         "use_kb": True
#     }
#     stream_chat(payload)

#     print("\n=== Test /chat with top_k override (use_kb=True) ===")
#     payload = {
#         "query": "Bar",
#         "session_id": "s2",
#         "kb_id": kb_id,
#         "top_k": 1,
#         "use_kb": True
#     }
#     stream_chat(payload)

# # 8. Test student CRUD tools (streaming)
# def test_student_crud():
#     sess = "student_crud"

#     print("\n=== Tool Call: create_student Frank ===")
#     create_args = {"name":"Frank","chinese":91,"math":82,"english":88}
#     stream_chat({"query": json.dumps(create_args), "session_id": sess, "use_kb": False})

#     print("\n=== Tool Call: read_student Frank ===")
#     stream_chat({"query": json.dumps({"name":"Frank"}), "session_id": sess, "use_kb": False})

#     print("\n=== Tool Call: update_student Frank ===")
#     update_args = {"name":"Frank","chinese":95,"math":90,"english":92}
#     stream_chat({"query": json.dumps(update_args), "session_id": sess, "use_kb": False})

#     print("\n=== Tool Call: read_student Frank (after update) ===")
#     stream_chat({"query": json.dumps({"name":"Frank"}), "session_id": sess, "use_kb": False})

#     print("\n=== Tool Call: delete_student Frank ===")
#     stream_chat({"query": json.dumps({"name":"Frank"}), "session_id": sess, "use_kb": False})

#     print("\n=== Tool Call: read_student Frank (after delete) ===")
#     stream_chat({"query": json.dumps({"name":"Frank"}), "session_id": sess, "use_kb": False})

# # 9. Test multi-turn dialogue both modes (streaming)
# def test_multiturn(kb_id):
#     print("\n=== Multi-turn with KB ===")
#     sess = "s4"
#     for msg in ["你好","请问你能做什么？","谢谢"]:
#         stream_chat({"query": msg, "session_id": sess, "kb_id": kb_id, "use_kb": True})

#     print("\n=== Multi-turn without KB ===")
#     sess2 = "s4_nokb"
#     for msg in ["你好","请问你能做什么？","谢谢"]:
#         stream_chat({"query": msg, "session_id": sess2, "use_kb": False})

# if __name__ == "__main__":
#     # 检查服务
#     check_service("Embedding Service", EMBED_URL + "/docs")
#     check_service("Chat Service",     CHAT_URL + "/docs")

#     test_embedding()
#     test_default_chat()
#     test_chat_without_kb()

#     kb_id = test_upload_kb()
#     if kb_id:
#         test_weights_and_topk(kb_id)
#         test_student_crud()
#         test_multiturn(kb_id)
#     else:
#         print("⚠️ 未能上传新 KB，跳过后续测试。")


import json

import pytest
# Assuming your FastAPI `app` is defined in a module named `client_customer_service.py`
from client_customer_service import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.mark.parametrize(
    "query, expected_tool",
    [
        ("查询学生 Alice 的成绩", "read_student"),
        ("添加学生 Zoe, 语文 82, 数学 90, 英语 88", "create_student"),
        ("把 Bob 的成绩改成 语文 80, 数学 85, 英语 90", "update_student"),
        ("删除学生 Diana", "delete_student"),
        ("列出所有学生", "list_students"),
    ],
)
def test_tool_call_and_result(query, expected_tool):
    payload = {
        "query": query,
        "session_id": "test_session",
        "kb_id": "default",
        "w_query": 0.9,
        "w_doc": 0.1,
        "top_k": 5,
        "use_kb": False,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    content = response.text

    # 确认有工具调用标签
    assert "<function_call>" in content
    # 确认调用的是预期的工具
    assert f'"name":"{expected_tool}"' in content

    # 确认有工具返回标签
    assert "<function_result>" in content
    # 提取工具返回结果 JSON 并校验格式
    result_json = content.split("<function_result>")[1].split("</function_result>")[0]
    result = json.loads(result_json)
    assert result["status"] in ("success", "fail")
    # 如果是 read_student 或 list_students，还要有具体数据字段
    if expected_tool == "read_student":
        assert "student" in result
    if expected_tool == "list_students":
        assert "students" in result


if __name__ == "__main__":
    pytest.main(["-q", "--disable-warnings", "--maxfail=1"])
