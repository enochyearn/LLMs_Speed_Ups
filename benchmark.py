import mlx.core as mx
import argparse
from decoding import naive_generate, generate
from llama import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Different Speed Ups Performance")

    parser.add_argument(
        "--model-path",
        help="Path to the model weights and tokenizer",
        default="mlx_model",
    )
    parser.add_argument(
        "--draft-model-path",
        help="Path to the model weights and tokenizer",
        default="mlx_model",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model. Ignored when --few-shot is provided.",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)


    draft_model, draft_tokenizer = load_model(args.draft_model_path)

    model, tokenizer = load_model(args.model_path)


    print("Naive Text Generation")
    naive_generate(args.prompt, tokenizer, model, temp=args.temp, max_tokens=args.max_tokens, write_every=args.write_every)
    print("------")
    print("KV Caching")
    generate(args.prompt, tokenizer, model, temp=args.temp, max_tokens=args.max_tokens, write_every=args.write_every)