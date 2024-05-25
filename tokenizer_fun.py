def tokenizer_char(text):
# build the encoder and decoder of tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(ch, 1) for ch in s] # encoder, convert a string into token ids (a list of integers)
    decode = lambda l: ''.join([itos[i] for i in l]) # decode, convert a list a integers into string
    return encode, decode, vocab_size