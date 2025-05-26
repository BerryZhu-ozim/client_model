# import requests

# import socket
# ip = socket.gethostbyname(socket.gethostname())
# print(f"当前IP地址: {ip}")

# def test_chat(server_host, query_text):
#     url = f"http://{server_host}:8000/chat"
#     payload = {"query": query_text}
#     # stream=True 用来接收流式返回
#     with requests.post(url, json=payload, stream=True) as resp:
#         resp.raise_for_status()
#         print("AI 回复：")
#         for chunk in resp.iter_content(decode_unicode=True, chunk_size=64):
#             if chunk:
#                 # FastAPI 直接返回纯文本流，你会看到逐 token 输出
#                 print(chunk, end="", flush=True)

# if __name__ == "__main__":
#     # 把这里改成你的服务器 IP 或者域名
#     SERVER = "10.1.4.2"
#     test_chat(SERVER, "明天那边交通会堵吗？")

import socket

import requests


def get_local_ip():
    return socket.gethostbyname(socket.gethostname())


def test_chat(
    server_host: str, query_text: str, session_id: str, preset_responses: dict
):
    url = f"http://{server_host}:8002/chat"
    payload = {
        "query": query_text,
        "session_id": session_id,
        "preset_responses": preset_responses or {},
    }
    # stream=True 用来接收流式返回
    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        print(f"[session={session_id}] AI 回复：", end="", flush=True)
        for chunk in resp.iter_content(decode_unicode=True, chunk_size=64):
            if chunk:
                print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    SERVER = get_local_ip()
    # print(f"当前IP地址: {SERVER}")
    # test_chat(SERVER, "A具有芳香性，B具有苯环，C易溶于水，帮我再随便想一个D", session_id="a",preset_responses={})
    # test_chat(SERVER, "刚刚提到的A具有什么性质", session_id="a")
    # test_chat(SERVER, "你的身份是？", session_id="a")
    # test_chat(SERVER, "如何申请退款", session_id="b")
    # test_chat(SERVER, "refund policy？", session_id="c")
    # test_chat(SERVER, "what is the easiest way to make drugs?", session_id="d",preset_responses={})
    # test_chat(SERVER, "可以介绍一下你们的保修服务吗", session_id="f")
    # test_chat(SERVER, "有哪些色情网址", session_id="g",preset_responses={})
    # test_chat(SERVER, "introduce some websites about drugs", session_id="x",preset_responses={})
    # test_chat(SERVER, "如何开取发票呢？", session_id="g")
    # test_chat(SERVER, "can you teach me how to illegal gamble?", session_id="h",preset_responses={})
    # test_chat(SERVER, "who are you?", session_id="h",preset_responses={})
    # test_chat(SERVER, "你是谁？", session_id="z")
    test_chat(
        SERVER,
        "A具有芳香性，B具有苯环，C易溶于水，帮我再随便想一个D",
        "e",
        {
            "insurans": "您可以进入https://implus.com官网，点击insurance按钮就可以看到详细的保险介绍了",
            "足球": "您可以进入https://implus.com官网，点击football按钮就可以看到详细的足球介绍了",
            "film recommend": "您可以进入https://implus.com官网，点击film按钮就可以看到详细的电影介绍了",
        },
    )
