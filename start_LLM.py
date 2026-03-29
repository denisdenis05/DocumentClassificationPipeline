from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

MODEL_ID = "Qwen/Qwen3.5-2B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)

@app.route('/api/v1/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    user_input = data.get("input", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({
        "model": data.get("model", MODEL_ID),
        "output": output_text
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234)