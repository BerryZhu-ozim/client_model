import re

def detect_language(text: str) -> str:
    # 先检测中文字符
    if re.search(r"[\u4e00-\u9fff]", text):
        return "中文"
    
    # 再检测 Malay 特征： -kan 结尾的动词 或者 一些高频功能词
    if re.search(r"\b\w+kan\b", text, re.IGNORECASE) \
       or re.search(r"\b(?:apa|nak|saya|boleh|macam|dan|yang|untuk|dengan|kepada|atau|kerana)\b", 
                    text, re.IGNORECASE):
        return "Malay"
    
    # 默认其它都当 English
    return "English"


if __name__ == "__main__":
    text = "Memperkenalkan Kepintaran Buatan"
    print(detect_language(text))