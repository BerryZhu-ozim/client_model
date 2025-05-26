# test_sglang_fixed.py
import os
import sys
import time

from openai import OpenAI  # type: ignore


def main():
    client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30001/v1")

    # 流式调用
    print("流式输出：")
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "介绍一下你自己"}],
        temperature=0.0,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        # 方法一：直接取属性
        content = chunk.choices[0].delta.content
        # 方法二：更通用的 getattr
        # content = getattr(chunk.choices[0].delta, "content", None)

        if content:
            print(content, end="", flush=True)
    print()  # 换行


if __name__ == "__main__":
    main()
