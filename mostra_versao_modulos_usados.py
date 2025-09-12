import importlib

libs = {
    "os": "os",
    "subprocess": "subprocess",
    "torch": "torch",
    "datasets": "datasets",
    "transformers": "transformers",
    "peft": "peft",
    "huggingface_hub[hf_xet]": "huggingface_hub[hf_xet]",
    "huggingface_hub": "huggingface_hub",
    "bitsandbytes": "bitsandbytes",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "sentencepiece": "sentencepiece",
    "mistral_common": "mistral_common",
}
for nome, modulo in libs.items():
    try:
        lib = importlib.import_module(modulo)
        versao = getattr(lib, "__version__", "Versão não disponível")
        print(f"{nome}: {versao}")
    except ImportError:
        print(f"{nome}: não instalado")