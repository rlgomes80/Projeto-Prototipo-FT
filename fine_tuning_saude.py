# -*- coding: utf-8 -*-
# fine_tuning_saude.py

def main():
    """
    Script principal para realizar o fine-tuning do modelo
    microsoft/Phi-3-mini-4k-instruct com um dataset de saúde do Brasil,
    utilizando a técnica de LoRA e salvando o resultado final em formato GGUF.
    """
    # Importação de bibliotecas essenciais.
    # Colocadas dentro da função main para encapsular todo o processo.
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

    # --- 1. Definição de Variáveis e Configurações Iniciais ---
    print("Iniciando o processo de fine-tuning...")

    # Nome do modelo base a ser utilizado do Hugging Face.
    NOME_MODELO_BASE = "microsoft/Phi-3-mini-4k-instruct"

    # Nome do arquivo de dataset no formato JSONL.
    NOME_ARQUIVO_DATASET = "fine_tuning_saude_2000.jsonl"

    # Diretórios para salvar os artefatos do treinamento.
    DIR_SAIDA_ADAPTADOR_LORA = "output_hf_model"
    DIR_SAIDA_MODELO_FINAL = "output_hf_model-final"
    
    # Nome do arquivo GGUF final.
    NOME_MODELO_GGUF = "novo_modelo.gguf"

    # --- 2. Configuração de Quantização (BitsAndBytes) ---
    # Configura o modelo para ser carregado em 4-bit, economizando memória VRAM.
    # bnb_4bit_quant_type='nf4': Usa o tipo de quantização Normal Float 4.
    # bnb_4bit_compute_dtype=torch.bfloat16: Tipo de dado para computação.
    # bnb_4bit_use_double_quant=True: Ativa a quantização dupla para maior economia.
    config_quantizacao = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # --- 3. Carregamento do Tokenizador e Modelo Base ---
    print(f"Carregando o modelo base: {NOME_MODELO_BASE}")

    # Carrega o tokenizador associado ao modelo.
    # trust_remote_code=True é necessário para modelos mais recentes como o Phi-3.
    tokenizador = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=NOME_MODELO_BASE,
        trust_remote_code=True
    )
    # Define o token de padding, caso não exista. Essencial para o treinamento.
    tokenizador.pad_token = tokenizador.eos_token
    tokenizador.padding_side = "right" # Garante que o padding seja adicionado à direita.

    # Carrega o modelo base com as configurações de quantização.
    # device_map="auto" distribui o modelo automaticamente pela GPU disponível.
    modelo_base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=NOME_MODELO_BASE,
        quantization_config=config_quantizacao,
        device_map="auto",
        trust_remote_code=True
    )

    # --- 4. Configuração e Aplicação do LoRA (PEFT) ---
    print("Configurando e aplicando o adaptador LoRA...")

    # Prepara o modelo quantizado para o treinamento com k-bit.
    modelo_base = prepare_model_for_kbit_training(model=modelo_base)

    # Configuração do LoRA.
    # r: O rank da matriz de atualização, controla o número de parâmetros treináveis.
    # lora_alpha: Fator de escala para as matrizes LoRA.
    # target_modules: Aplica LoRA em todas as camadas lineares do modelo.
    # lora_dropout: Dropout para as camadas LoRA, previne overfitting.
    # bias="none": Não treina os bias, prática comum com LoRA.
    # task_type="CAUSAL_LM": Especifica a tarefa como modelagem de linguagem causal.
    config_lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules='all-linear',
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Aplica as configurações LoRA ao modelo base.
    modelo_peft = get_peft_model(
        model=modelo_base, 
        peft_config=config_lora
    )
    
    # Imprime o número de parâmetros que serão treinados.
    modelo_peft.print_trainable_parameters()

    # --- 5. Carregamento e Preparação do Dataset ---
    print(f"Carregando e processando o dataset: {NOME_ARQUIVO_DATASET}")
    
    # Carrega o dataset a partir do arquivo JSONL local.
    dataset = load_dataset(
        path='json', 
        data_files=NOME_ARQUIVO_DATASET, 
        split='train'
    )

    # Função para formatar e tokenizar os dados.
    # O formato segue o padrão de instrução do Phi-3.
    def formatar_e_tokenizar(exemplo):
        prompt_formatado = f"<|user|>\n{exemplo['input']}<|end|>\n<|assistant|>\n{exemplo['output']}<|end|>"
        # Tokeniza o prompt formatado.
        resultado = tokenizador(
            prompt_formatado,
            truncation=True,
            max_length=512, # Limita o tamanho para evitar consumo excessivo de memória.
            padding="max_length"
        )
        # O modelo precisa dos 'labels' para calcular a loss.
        # Em Causal LM, os labels são os próprios input_ids.
        resultado["labels"] = resultado["input_ids"][:]
        return resultado

    # Aplica a função de formatação a todo o dataset.
    dataset_tokenizado = dataset.map(
        function=formatar_e_tokenizar
    )

    # --- 6. Configuração dos Argumentos de Treinamento ---
    print("Definindo os argumentos de treinamento...")
    
    args_treinamento = TrainingArguments(
        output_dir=DIR_SAIDA_ADAPTADOR_LORA,
        num_train_epochs=3,
        per_device_train_batch_size=1, # Batch size pequeno para caber na VRAM.
        gradient_accumulation_steps=4, # Simula um batch size maior (1*4=4).
        optim="paged_adamw_8bit", # Otimizador eficiente em memória.
        logging_steps=25,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True, # Usa precisão mista para acelerar o treino e economizar memória.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, # Conforme solicitado.
        push_to_hub=False
    )
    
    # --- 7. Treinamento do Modelo ---
    # Collator para agrupar os dados em lotes. mlm=False para Causal LM.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizador, 
        mlm=False
    )

    # Inicializa o Trainer.
    trainer = Trainer(
        model=modelo_peft,
        args=args_treinamento,
        train_dataset=dataset_tokenizado, # type: ignore
        data_collator=data_collator
    )

    # Inicia o treinamento.
    print("Iniciando o treinamento do modelo...")
    trainer.train()
    print("Treinamento concluído.")

    # Salva o adaptador LoRA treinado e a configuração.
    print(f"Salvando o adaptador LoRA em '{DIR_SAIDA_ADAPTADOR_LORA}'...")
    trainer.save_model()
    tokenizador.save_pretrained(save_directory=DIR_SAIDA_ADAPTADOR_LORA)

    # --- 8. Merge do Modelo Base com o Adaptador LoRA ---
    print("Realizando o merge do modelo base com o adaptador LoRA...")

    # Limpa a memória da GPU antes de carregar o modelo novamente.
    del modelo_base
    del modelo_peft
    del trainer
    torch.cuda.empty_cache()

    # Carrega o modelo base novamente.
    modelo_base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=NOME_MODELO_BASE,
        low_cpu_mem_usage=True,
        return_dict=True,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Carrega o adaptador LoRA e o mescla com o modelo base.
    modelo_final = PeftModel.from_pretrained(
        model=modelo_base,
        model_id=DIR_SAIDA_ADAPTADOR_LORA
    )
    modelo_final = modelo_final.merge_and_unload() # type: ignore

    # Salva o modelo final mesclado.
    print(f"Salvando o modelo final mesclado em '{DIR_SAIDA_MODELO_FINAL}'...")
    modelo_final.save_pretrained(
        save_directory=DIR_SAIDA_MODELO_FINAL, 
        safe_serialization=True,
        max_shard_size="2GB" # Divide o modelo em partes de no máximo 2GB.
    )
    # Salva o tokenizador junto com o modelo final.
    tokenizador.save_pretrained(save_directory=DIR_SAIDA_MODELO_FINAL)
    
    # --- 9. Conversão para o Formato GGUF ---
    print("Iniciando a conversão para o formato GGUF...")
    
    # Define o caminho para o script de conversão do llama.cpp.
    # Assume que o repositório llama.cpp está no mesmo diretório do script.
    caminho_script_gguf = os.path.join("llama.cpp", "convert_hf_to_gguf.py")

    if not os.path.exists(caminho_script_gguf):
        print("ERRO: O script 'convert_hf_to_gguf.py' não foi encontrado.")
        print("Certifique-se de que o repositório 'llama.cpp' foi clonado no diretório atual.")
        return

    # Constrói e executa o comando de conversão.
    # --outtype q8_0: Quantização de 8 bits, bom equilíbrio entre tamanho e performance.
    comando_conversao = [
        "python",
        caminho_script_gguf,
        DIR_SAIDA_MODELO_FINAL,
        "--outtype",
        "q8_0",
        "--outfile",
        NOME_MODELO_GGUF
    ]
    
    try:
        subprocess.run(comando_conversao, check=True)
        print(f"\nModelo convertido com sucesso para '{NOME_MODELO_GGUF}'.")
    except subprocess.CalledProcessError as e:
        print(f"\nOcorreu um erro durante a conversão para GGUF: {e}")
    except FileNotFoundError:
        print("\nERRO: O comando 'python' não foi encontrado. Verifique sua instalação do Python e o PATH do sistema.")

    print("\nProcesso de fine-tuning e conversão finalizado!")


# Ponto de entrada do script.
if __name__ == "__main__":
    main()