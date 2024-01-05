import mlx.core as mx
from llama import load_model
import time

def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"

# Naive Text Generation, without any speed ups
def naive_generate(prompt, tokenizer, model, temp=1.0, max_tokens=40, write_every=1):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))
    print("------")
    print(prompt)
    tokenized_prompt = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    while True:
        x = mx.array([tokenized_prompt + tokens])
        logits = model(x)
        token = sample(logits[:,-1])
        mx.eval(token)
        tokens.append(token.item())

        if len(tokens) == 1:            
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= max_tokens:
            break

        elif (len(tokens) % write_every) == 0:
            s = tokenizer.decode([t for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    # mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)

def generate(prompt, tokenizer, model, temp=1.0, max_tokens=40, write_every=1):
    print("------")
    print(prompt)
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(x, temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= max_tokens:
            break

        elif (len(tokens) % write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)