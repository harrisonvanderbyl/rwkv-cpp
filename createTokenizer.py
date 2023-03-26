from pytext.torchscript.tokenizer.bpe import ScriptBPE

scbpe = ScriptBPE("./tokens.txt")

t = scbpe.tokenize("Hello World")

print(t)

