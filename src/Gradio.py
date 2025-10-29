import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft.peft_model import PeftModel

    
# load model and tokenizer
model_name = 'meta-llama/Llama-3.2-1B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)

model.to(dtype).to(device)

model.eval()

def respond(message, system_message, max_tokens, temperature, top_p):
    prompt = f"{system_message.strip()}\nquestion: {message.strip()}\nresponse:"
    print("=== PROMPT ===")
    print(prompt)

    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=1.1,
        num_beams=1
    )

    full_output = tok.decode(outputs[0], skip_special_tokens=True)
    print("=== OUTPUT ===")
    print(full_output)

    if "response:" in full_output:
        answer = full_output.split("response:", 1)[-1].strip()
    else:
        answer = full_output.strip()

    for stop_token in ["question:", "#", "\n\n", "##", "Chapter", "def ", "print("]:
        if stop_token in answer:
            answer = answer.split(stop_token, 1)[0].strip()

    return answer if answer else "Sorry, I haven't found the answer yet."


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="", label="(Optional) System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p (nucleus sampling)")
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
