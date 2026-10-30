import json, os
from openai import OpenAI
from detector import CognitionDetector

LOCAL = OpenAI(base_url='http://127.0.0.1:8080/v1', api_key='sk-no-key')
REMOTE = None
DET = CognitionDetector()

if os.path.exists("api.json"):
    try:
        with open('api.json', 'r', encoding='utf-8') as f:
            CFG = json.load(f)
            if CFG.get('baseURL'):
                REMOTE = OpenAI(base_url=CFG['baseURL'], api_key=CFG.get('apiKey', ''))
                DET = CognitionDetector(CFG.get('highThreshold', -1.5), CFG.get('lowThreshold', -3.5))
                print(f"✅ 已加载 api.json，配置接口：{CFG['baseURL']}")
    except Exception as e:
        print(f"⚠️ api.json 读取失败，保持纯离线运行: {e}")
else:
    print("💡 未发现 api.json，模型将完全断网离线运行。")

def chat(q):
    prompt = f"<|user|>\n{q}\n<|assistant|>\n"
    r = LOCAL.chat.completions.create(
        model='local', messages=[{'role': 'user', 'content': prompt}],
        logprobs=True, top_logprobs=5, max_tokens=128, temperature=0.1
    )
    a = r.choices[0].message.content
    band, s = DET.band(r.choices[0].logprobs)
    
    if band == 'LOW':
        if REMOTE:
            return REMOTE.chat.completions.create(
                model=CFG.get('remoteModel', 'gpt-4o-mini'),
                messages=[{'role': 'user', 'content': q}]
            ).choices[0].message.content
        else:
            return '这个问题超出了当前模型的认知范围。'
            
    if band == 'MID' and REMOTE:
        return REMOTE.chat.completions.create(
            model=CFG.get('remoteModel', 'gpt-4o-mini'),
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content
        
    return a
