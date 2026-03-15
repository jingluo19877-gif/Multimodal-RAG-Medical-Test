from ollama import Client

# 1. 替换为你刚才在 Colab 运行出来的那个随机 ngrok 地址
REMOTE_URL = 'https://mistakenly-hungry-tonie.ngrok-free.dev'

# 2. 初始化客户端，指向云端
client = Client(host=REMOTE_URL)


def test_medical_vision():
    # 随便找一张本地测试用的医学图片（比如网上下个 X 光片存为 test.jpg）
    image_path = r"D:\TestPicture\1.png"

    print(f"正在连接云端大模型 {REMOTE_URL} ...")

    # 3. 发送请求给 Colab 上的模型
    response = client.chat(
        model='dcarrascosa/medgemma-1.5-4b-it:Q8_0',
        messages=[{
            'role': 'user',
            'content': '请用中文分析这张医学影像，指出可能的异常区域。',
            'images': [image_path]
        }]
    )

    print("\n🩺 AI 诊断结果：")
    print(response['message']['content'])


if __name__ == '__main__':
    test_medical_vision()