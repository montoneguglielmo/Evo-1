if __name__ == "__main__":

    
    # In this file we load the tokenizer from the Hugging Face Hub and look into the files that define the tokenizer
    # The tokenizer would do the following steps:
    # 1. Text normalization: lowercase, remove accents, etc.
    # 2. Pre-tokenization: split sentences into words
    # 3. Tokenization: split words into tokens (BPE, WordPiece, etc.)
    # 4. Mapping tokens to ids: each token is mapped to a unique id
    # 5. Adding special tokens that helps models know where the text starts and ends
    # 6. Padding and truncation: pad the input to the same length as the model's input length
    # 7. Attention mask: create a mask to indicate which tokens are padding tokens

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B")
    print('Tokenizer loaded')
    print(type(tokenizer))
    print('='*100)
    
    # Download the tokenizer from the Hugging Face Hub
    from huggingface_hub import snapshot_download
    path_tokenizer = snapshot_download(
        repo_id="OpenGVLab/InternVL3-1B",
        allow_patterns=["tokenizer*"]
    )
    print('Tokenizer downloaded in the folder')
    print(path_tokenizer)
    print('='*100)
    
    # List the files in the tokenizer folder
    import os
    print('Files in the tokenizer folder:')
    for f in os.listdir(path_tokenizer):
        print(f)
    print('='*100)
    
    # Load the vocab file
    print('Vocab file: mapping of tokens to ids')
    import json
    import random
    with open(os.path.join(path_tokenizer, "vocab.json"), "r") as f:
        vocab = json.load(f)
    
    # Print some random entries from the vocabulary
    num_samples = min(100, len(vocab))  # Sample up to 20 entries
    random_entries = random.sample(list(vocab.items()), num_samples)
    print(f"Showing {num_samples} random entries from vocabulary (total size: {len(vocab)}):")
    for token, token_id in random_entries:
        print(f"  '{token}': {token_id}")
    print('='*100)
    
    
    # Test sentences
    texts = [
        "Hello world!",
        "Tokenization is very important for large language models.",
    ]

    # -----------------------------
    # 1. Basic tokenization
    # -----------------------------
    encoded = tokenizer(texts)
    print("Basic encoding output:")
    for i, e in enumerate(encoded["input_ids"]):
        print(f"Sentence {i} length:", len(e))
    print("=" * 100)

    # -----------------------------
    # 2. Show tokens and IDs
    # -----------------------------
    tokens = tokenizer.tokenize(texts[1])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("Tokens:")
    print(tokens)
    print("Token IDs:")
    print(token_ids)
    print("=" * 100)

    # -----------------------------
    # 3. Decode back to text
    # -----------------------------
    decoded = tokenizer.decode(token_ids)
    print("Decoded text:")
    print(decoded)
    print("=" * 100)

    # -----------------------------
    # 4. Padding & truncation test
    # -----------------------------
    encoded_padded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors="pt"
    )

    print("Padded & truncated encoding:")
    for k, v in encoded_padded.items():
        print(f"{k}:")
        print(v)
    print("=" * 100)

    # -----------------------------
    # 5. Special tokens inspection
    # -----------------------------
    print("Special tokens:")
    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("UNK:", tokenizer.unk_token, tokenizer.unk_token_id)
    print("=" * 100)