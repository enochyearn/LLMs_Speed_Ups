import mlx.core as mx
from llama import load_model
import time

def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"

# Naive Text Generation, without any speed ups
# Aka, Autoregressive Sampling, Autoregressive Decoding
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

def speculative_decoding(prompt, tokenizer, model, draft_model, temp=1.0, max_tokens=40, write_every=1, n_draft = 8):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))
    def max_fn(x):
        x_max = mx.where(x>0, x, 0)
        return x_max/mx.sum(x_max)
    def flush_print(end_char=""):
        nonlocal skip
        nonlocal tokens
        nonlocal write_every
        if (len(tokens) % write_every) == 0:
            s = tokenizer.decode([t for t in tokens])
            print(s[skip:], end=end_char, flush=True)
            skip = len(s)
    print("------")
    print(prompt)
    tokenized_prompt = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    n = len(tokenized_prompt)
    T = len(tokenized_prompt) + max_tokens
    skip = 0
    tokens = []
    start = tic()
    while n < T:
        draft = tokenized_prompt + tokens
        draft_tokens = []
        for _ in range(n_draft):
            draft_x = mx.array([draft + draft_tokens])
            draft_logits = draft_model(draft_x)
            draft_token = sample(draft_logits[:, -1])
            mx.eval(draft_token)
            draft_tokens.append(draft_token.item())

        draft = draft + draft_tokens
        x = mx.array([draft])
        logits = model(x)
        mx.eval(logits)

                
        all_accepted = True
        for _ in range(n_draft):
            i = n -1
            j = draft[i+1]
            selected_logits, selected_draft_logits = logits[:, i], draft_logits[:, i]
            logits_ratio = min(1, (selected_logits[:, j]/selected_draft_logits[:, j]).item())

            if n >= T:
                all_accepted = False
                break
            elif (mx.random.uniform().item() < logits_ratio):
                tokens.append(j)
                n += 1
                flush_print()
            else:
                resampled_token = sample(selected_logits)
                mx.eval(resampled_token)
                tokens.append(resampled_token.item())
                n += 1
                all_accepted = False
                flush_print()
                break
            if len(tokens) == 1:            
                prompt_processing = toc("Prompt processing", start)

        if all_accepted:
            last_token = sample(logits[:, -1])
            mx.eval(last_token)
            tokens.append(last_token.item())
            n += 1 
            if (len(tokens) % write_every) == 0:
                s = tokenizer.decode([t for t in tokens])
                print(s[skip:], end="", flush=True)
                skip = len(s)
    full_gen = toc("Full generation", start)
    flush_print(end_char="\n")
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)