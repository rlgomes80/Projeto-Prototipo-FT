# Script de Fine Tuning (FT) de Small Language Model (SLM) para Saúde

## Visão Geral

Este projeto implementa uma solução completa para o fine-tuning de um Small Language Model (SLM), especificamente o `microsoft/Phi-3-mini-4k-instruct`, especializado em terminologias e contextos da Atenção Primária à Saúde (APS) no Brasil. 

A solução utiliza técnicas avançadas de otimização:
- **Low-Rank Adaptation (LoRA)** para treinamento eficiente com recursos limitados
- **Conversão para formato GGUF** para inferência otimizada em CPUs e GPUs
- **Compatibilidade com llama.cpp** para deployment em ambientes com hardware restrito
- **Integração preparada para sistemas RAG** (Retrieval-Augmented Generation)

## Especificações Técnicas

### Ambiente de Desenvolvimento
- **Sistema Operacional**: Windows 11
- **Memória RAM**: 32GB
- **GPU**: NVIDIA GeForce RTX 5060 (8GB VRAM)
- **Arquitetura**: x64

### Autor
**Ricardo Gomes**  
Desenvolvido utilizando técnicas de Engenharia de Prompts e Vibe Coding.

## Pré-requisitos do Sistema

### 1. CUDA Toolkit
Instalação do ambiente de desenvolvimento NVIDIA CUDA:
```
Versão: CUDA 12.9.0
Arquivo: cuda_12.9.0_576.02_windows.exe
Download: https://developer.nvidia.com/cuda-12-9-0-download-archive
```

### 2. CMake
Sistema de build multiplataforma:
```
Recomendado: Versão estável mais recente
Arquivo: cmake-x.x.x-windows-x86_64.msi
Download: https://cmake.org/download/
```

### 3. Visual Studio Build Tools
Compilador Microsoft Visual C++:
```
Produto: Visual Studio Community 2022
Componente obrigatório: "Desenvolvimento para desktop com C++"
Download: https://visualstudio.microsoft.com/pt-br/vs/community/
```

**Nota importante**: Durante a instalação, certifique-se de selecionar a carga de trabalho "Desenvolvimento para desktop com C++" para incluir o compilador MSVC e todas as ferramentas de build necessárias.

### 4. Git
Sistema de controle de versão:
```
Versão: 2.51.0 ou superior
Arquivo: Git-x.x.x-64-bit.exe
Download: https://git-scm.com/downloads/win
```

## Instalação e Configuração

### Compilação do llama.cpp

Execute os seguintes comandos no Windows PowerShell como Administrador:

```powershell
# Navegue até o diretório do projeto
cd <caminho_do_projeto>

# Clone o repositório oficial
git clone https://github.com/ggerganov/llama.cpp.git

# Acesse o diretório
cd llama.cpp

# Inicialize submódulos
git submodule update --init --recursive

# Configure o ambiente de build
mkdir build
cd build

# Configure com suporte CUDA
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON

# Compile em modo Release
cmake --build . --config Release
```

**Observação**: Se encontrar erros de compilação, execute os comandos no "x64 Native Tools Command Prompt for VS" para garantir que as variáveis de ambiente do Visual Studio estejam carregadas corretamente.

### Configuração do Ambiente Python

#### Windows (PowerShell/CMD)
```cmd
# Criar ambiente virtual isolado
python -m venv .venv

# Ativar ambiente virtual
.\.venv\Scripts\activate

# Atualizar gerenciador de pacotes
python -m pip install --upgrade pip
```

#### Linux (Terminal)
```bash
# Instalar dependências do sistema
sudo apt install python3 python3-pip python3.10-venv

# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate

# Atualizar pip
python3 -m pip install --upgrade pip
```

### Instalação das Dependências Python

#### 1. PyTorch com Suporte CUDA
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

**Nota técnica**: Utilizamos a build CUDA 12.9 (cu129) para CUDA 12.9. Esta versão é totalmente compatível com drivers NVIDIA mais recentes e CUDA Toolkit 12.x.

#### 2. Bibliotecas de Machine Learning e NLP
```bash
pip install huggingface_hub[hf_xet] transformers datasets bitsandbytes peft accelerate safetensors sentencepiece mistral_common
```

## Versões das Dependências

### Bibliotecas Principais
| Biblioteca | Versão | Função |
|------------|--------|---------|
| `torch` | 2.8.0+cu129 | Framework de deep learning |
| `transformers` | 4.56.1 | Modelos de linguagem pré-treinados |
| `datasets` | 4.0.0 | Manipulação de datasets |
| `peft` | 0.17.1 | Parameter-Efficient Fine-Tuning |
| `accelerate` | 1.10.1 | Treinamento distribuído |
| `bitsandbytes` | 0.47.0 | Quantização e otimização |

### Bibliotecas de Suporte
| Biblioteca | Versão | Função |
|------------|--------|---------|
| `huggingface_hub` | 0.34.4 | Integração com Hugging Face Hub |
| `safetensors` | 0.6.2 | Serialização segura de tensores |
| `sentencepiece` | 0.2.1 | Tokenização |
| `mistral_common` | 1.8.4 | Utilitários para modelos Mistral |

## Instalando Modelo Treinado no Ollama
```cmd
ollama create Phi-3-mini-4k-instruct-APS -f Modelfile
```

**ATENÇÃO:** Lembre-se de ajustar o arquivo `Modelfile` conforme os parâmetros do `.env`

## Características do Projeto

### Otimizações Implementadas
- **Eficiência de Memória**: Uso de LoRA para reduzir requisitos de VRAM
- **Performance**: Compilação otimizada com CUDA para aceleração GPU
- **Portabilidade**: Conversão GGUF para deployment em diferentes arquiteturas
- **Especialização**: Fine-tuning específico para terminologia médica brasileira

### Casos de Uso
- Sistemas de apoio à decisão clínica
- Assistentes virtuais para profissionais de saúde
- Ferramentas de triagem e classificação médica
- Integração com prontuários eletrônicos
- Sistemas RAG para consulta de diretrizes clínicas

## Suporte

Para questões técnicas ou suporte:
- Entre em contato com o autor através dos canais oficiais
- Consulte a documentação das bibliotecas utilizadas

---

**Nota de Responsabilidade**: Este projeto destina-se exclusivamente a fins educacionais e de pesquisa. Para uso em ambiente clínico real, é necessária validação adicional e conformidade com regulamentações sanitárias aplicáveis.