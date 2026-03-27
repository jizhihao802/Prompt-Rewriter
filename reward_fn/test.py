import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/model/qwen2.5-0.5b",
    trust_remote_code=True
)

_model_local = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/model/qwen2.5-0.5b",
    device_map={"": device},
    dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

if __name__ == "__main__":
    # 编码输入
    instruction = "我希望你能担任咖啡聊天的冷邮件撰写人。我会提供收件人以及其他相关信息，你需要写出一封专业有效的冷邮件来请求一次咖啡聊天。咖啡聊天的目的是更好地了解收件人，可能建立联系，并可能寻求转介和建议。电子邮件应简明扼要，并清晰地概述咖啡聊天的目的以及收件人同意举行咖啡聊天所能得到的任何好处或价值。请不要包含任何个人意见或不必要的细节，并确保电子邮件的语气礼貌和尊重。电子邮件还应包括明确的呼吁收件人在方便的时间安排咖啡聊天。\n收件人是David Wan。这是他的领英关于部分：“我有兴趣利用先进的计算工具通过健壮的统计理论来分析大量数据，以生成可行的见解，并通过解决任何问题来产生真正的实际影响，以使人们的生活变得更好。”"

    prompt = (
        f"你是一个指令优化器，负责优化用户给到大模型的指令，请在保留信息完整的前提下按照按照缩短指令长度、调整指令结构的方法优化<instruction>和</instruction>之间的指令。"
        f"<instruction>\n{instruction}\n</instruction>"
        f"并按照“<result>你优化后的指令</result>”的格式输出结果，并在</result>后终止输出。"
    )
    prompt2 = f"你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令从<method>和</method>之间的优化方法中选择合适的一个或几个优化方式。<instruction>{instruction}</instruction>。<method>1.缩短指令长度 2.调整指令结构 3.增加任务说明</method>只输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，不输出其他任何内容。"
    inputs = _tokenizer(prompt2, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    max_new_tokens = 0

    if max_new_tokens == 0:
        max_new_tokens=len(prompt)

    with torch.no_grad():
        # 生成序列
        outputs = _model_local.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )

    # 解码整个输出（包含输入 + 新生成部分）
    full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ====== 去掉 prompt 对应部分，只保留新生成内容 ======
    prompt_text = _tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
    if full_text.startswith(prompt_text):
        new_text = full_text[len(prompt_text):].strip()
    else:
        new_text = full_text  # 如果没有匹配，直接返回全部
    
    print(f"new instruction:{new_text}")

