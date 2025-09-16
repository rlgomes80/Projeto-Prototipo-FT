# -*- coding: utf-8 -*-
# fine_tuning_saude.py

def main():
    """
    Função principal que executa o pipeline completo de fine-tuning do modelo Phi-3-mini-4k-instruct
    com dataset de saúde brasileiro, utilizando LoRA (Low-Rank Adaptation) e conversão final para GGUF.
    """
    # Importações das dependências necessárias - encapsuladas na main() para isolamento
    import os
    import subprocess
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        Trainer
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training
    )

    # --- 1. Inicialização de constantes e configuração base ---
    print("Iniciando o processo de fine-tuning...")

    # Identificador do modelo pré-treinado no Hugging Face Hub
    NOME_MODELO_BASE = "microsoft/Phi-3-mini-4k-instruct"

    # Arquivo local contendo o dataset em formato JSON Lines (um JSON por linha)
    NOME_ARQUIVO_DATASET = "glossario_SAUDE_e_perguntas_respostas_APS.jsonl"

    # Diretório onde o adaptador LoRA será salvo após treinamento
    DIR_SAIDA_ADAPTADOR_LORA = "output_hf_model"
    # Diretório para o modelo final após merge do base + adaptador
    DIR_SAIDA_MODELO_FINAL = "output_hf_model-final"
    
    # Nome do arquivo final no formato GGUF (formato otimizado para inferência)
    NOME_MODELO_GGUF = "novo_modelo.gguf"

    # --- 2. Configuração da quantização 4-bit para otimização de memória ---
    # BitsAndBytesConfig otimiza uso de VRAM através de quantização
    config_quantizacao = BitsAndBytesConfig(
        load_in_4bit=True,                          # Carrega modelo em precisão 4-bit
        bnb_4bit_quant_type="nf4",                 # Tipo Normal Float 4 (melhor qualidade)
        bnb_4bit_compute_dtype=torch.bfloat16,     # Tipo de dados para cálculos (bfloat16 é eficiente)
        bnb_4bit_use_double_quant=True             # Quantização dupla para maior compressão
    )

    # --- 3. Carregamento e configuração do tokenizador ---
    print(f"Carregando o modelo base: {NOME_MODELO_BASE}")

    # Carrega tokenizador específico do modelo Phi-3
    tokenizador = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=NOME_MODELO_BASE,
        trust_remote_code=True                      # Necessário para modelos com código customizado
    )
    # Define token de padding igual ao token de fim de sequência
    tokenizador.pad_token = tokenizador.eos_token
    # Configura padding à direita (padrão para modelos causais)
    tokenizador.padding_side = "right"

    # --- 4. Carregamento do modelo base com quantização ---
    modelo_base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=NOME_MODELO_BASE,
        quantization_config=config_quantizacao,    # Aplica config de quantização 4-bit
        device_map="auto",                         # Distribui camadas automaticamente entre devices
        trust_remote_code=True                     # Permite execução de código customizado do modelo
    )

    # --- 5. Preparação do modelo para treinamento k-bit e configuração LoRA ---
    print("Configurando e aplicando o adaptador LoRA...")

    # Prepara modelo quantizado para treinamento eficiente
    modelo_base = prepare_model_for_kbit_training(model=modelo_base)

    # Configuração dos hiperparâmetros do LoRA
    config_lora = LoraConfig(
        r=16,                                      # Rank da decomposição (controla parâmetros treináveis)
        lora_alpha=32,                             # Fator de escala das atualizações LoRA
        target_modules='all-linear',               # Aplica LoRA em todas camadas lineares do modelo
        lora_dropout=0.05,                         # Dropout para regularização (previne overfitting)
        bias="none",                               # Não treina biases (economia de parâmetros)
        task_type="CAUSAL_LM"                      # Especifica tarefa de modelagem de linguagem causal
    )

    # Aplica adaptador LoRA ao modelo base
    modelo_peft = get_peft_model(
        model=modelo_base, 
        peft_config=config_lora
    )
    
    # Exibe estatísticas de parâmetros treináveis vs totais
    modelo_peft.print_trainable_parameters()

    # --- 6. Carregamento e pré-processamento do dataset ---
    print(f"Carregando e processando o dataset: {NOME_ARQUIVO_DATASET}")
    
    # Carrega dataset local no formato JSON Lines
    dataset = load_dataset(
        path='json',                               # Loader para arquivos JSON/JSONL
        data_files=NOME_ARQUIVO_DATASET,          # Caminho do arquivo local
        split='train'                              # Define como split de treinamento
    )

    # Função de formatação e tokenização dos exemplos do dataset
    def formatar_e_tokenizar(exemplo):
        # Formata exemplo seguindo template de chat do Phi-3
        prompt_formatado = f"<|user|>\n{exemplo['input']}<|end|>\n<|assistant|>\n{exemplo['output']}<|end|>"
        # Tokeniza o texto formatado com limite de tokens
        resultado = tokenizador(
            prompt_formatado,
            truncation=True,                       # Trunca sequências longas
            max_length=512,                        # Limite máximo de tokens por exemplo
            padding="max_length"                   # Padding até comprimento máximo
        )
        # Cria labels idênticos aos input_ids para treinamento causal
        resultado["labels"] = resultado["input_ids"][:]
        return resultado

    # Aplica pré-processamento a todos exemplos do dataset
    dataset_tokenizado = dataset.map(
        function=formatar_e_tokenizar
    )

    # --- 7. Configuração dos parâmetros de treinamento ---
    print("Definindo os argumentos de treinamento...")
    
    # Argumentos que controlam o processo de treinamento
    args_treinamento = TrainingArguments(
        output_dir=DIR_SAIDA_ADAPTADOR_LORA,      # Diretório de saída do modelo treinado
        num_train_epochs=3,                        # Número de épocas de treinamento
        per_device_train_batch_size=1,            # Tamanho do batch por device (limitado por VRAM)
        gradient_accumulation_steps=4,             # Simula batch_size=4 através de acumulação
        optim="paged_adamw_8bit",                 # Otimizador AdamW com paginação 8-bit
        logging_steps=25,                          # Frequência de logging das métricas
        save_strategy="epoch",                     # Salva checkpoint a cada época
        learning_rate=2e-4,                        # Taxa de aprendizado
        fp16=True,                                 # Usa precisão mista FP16 para eficiência
        gradient_checkpointing=True,               # Economiza memória recalculando gradientes
        gradient_checkpointing_kwargs={'use_reentrant': False}, # Config para compatibilidade
        push_to_hub=False                          # Não faz upload automático para Hugging Face Hub
    )
    
    # --- 8. Execução do treinamento ---
    # Data collator para agrupamento de exemplos em batches
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizador, 
        mlm=False                                  # False para modelagem de linguagem causal
    )

    # Inicializa objeto Trainer com modelo, argumentos e dados
    trainer = Trainer(
        model=modelo_peft,
        args=args_treinamento,
        train_dataset=dataset_tokenizado,          # Dataset tokenizado para treinamento # type: ignore
        data_collator=data_collator
    )

    # Executa o loop de treinamento
    print("Iniciando o treinamento do modelo...")
    trainer.train()
    print("Treinamento concluído.")

    # Salva o adaptador LoRA e tokenizador
    print(f"Salvando o adaptador LoRA em '{DIR_SAIDA_ADAPTADOR_LORA}'...")
    trainer.save_model()
    tokenizador.save_pretrained(save_directory=DIR_SAIDA_ADAPTADOR_LORA)

    # --- 9. Merge do modelo base com adaptador LoRA ---
    print("Realizando o merge do modelo base com o adaptador LoRA...")

    # Limpeza de memória GPU antes de recarregar modelo
    del modelo_base
    del modelo_peft
    del trainer
    torch.cuda.empty_cache()

    # Recarrega modelo base sem quantização para merge
    modelo_base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=NOME_MODELO_BASE,
        low_cpu_mem_usage=True,                    # Otimiza uso de RAM durante carregamento
        return_dict=True,                          # Retorna dicionário em vez de tupla
        dtype=torch.float16,                       # Usa FP16 para economia de memória
        device_map="auto",                         # Distribuição automática entre devices
        trust_remote_code=True,
    )

    # Carrega adaptador LoRA treinado
    modelo_final = PeftModel.from_pretrained(
        model=modelo_base,
        model_id=DIR_SAIDA_ADAPTADOR_LORA          # Diretório contendo o adaptador salvo
    )
    # Mescla adaptador com modelo base, criando modelo unificado
    modelo_final = modelo_final.merge_and_unload() # type: ignore

    # Salva modelo final mesclado
    print(f"Salvando o modelo final mesclado em '{DIR_SAIDA_MODELO_FINAL}'...")
    modelo_final.save_pretrained(
        save_directory=DIR_SAIDA_MODELO_FINAL, 
        safe_serialization=True,                   # Usa formato Safetensors (mais seguro)
        max_shard_size="2GB"                       # Divide modelo em arquivos de máximo 2GB
    )
    # Salva tokenizador junto com modelo final
    tokenizador.save_pretrained(save_directory=DIR_SAIDA_MODELO_FINAL)
    
    # --- 10. Conversão para formato GGUF ---
    print("Iniciando a conversão para o formato GGUF...")
    
    # Caminho para script de conversão do repositório llama.cpp
    caminho_script_gguf = os.path.join("llama.cpp", "convert_hf_to_gguf.py")

    # Verifica se script de conversão existe
    if not os.path.exists(caminho_script_gguf):
        print("ERRO: O script 'convert_hf_to_gguf.py' não foi encontrado.")
        print("Certifique-se de que o repositório 'llama.cpp' foi clonado no diretório atual.")
        return

    # Constrói comando de conversão para GGUF
    comando_conversao = [
        "python",
        caminho_script_gguf,                       # Script de conversão HF -> GGUF
        DIR_SAIDA_MODELO_FINAL,                   # Diretório do modelo fonte
        "--outtype",
        "q8_0",                                    # Quantização 8-bit (balanceio tamanho/qualidade)
        "--outfile",
        NOME_MODELO_GGUF                          # Nome do arquivo GGUF de saída
    ]
    
    # Executa conversão com tratamento de erros
    try:
        subprocess.run(comando_conversao, check=True)
        print(f"\nModelo convertido com sucesso para '{NOME_MODELO_GGUF}'.")
    except subprocess.CalledProcessError as e:
        print(f"\nOcorreu um erro durante a conversão para GGUF: {e}")
    except FileNotFoundError:
        print("\nERRO: O comando 'python' não foi encontrado. Verifique sua instalação do Python e o PATH do sistema.")

    print("\nProcesso de fine-tuning e conversão finalizado!")


# Ponto de entrada - executa main() apenas se script for chamado diretamente
if __name__ == "__main__":
    main()