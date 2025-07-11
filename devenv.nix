{ pkgs, lib, config, inputs, ... }:

{
  languages = {
    shell.enable = true;
    nix.enable = true;
    python = {
      enable = true;
	    uv.enable = true;
	    venv = {
	    	enable = true;
		requirements = ''
regex # Replace re for higher-performance regex matching
cachetools
psutil
sentencepiece  # Required for LLaMA tokenizer.
numpy
requests >= 2.26.0
tqdm
blake3
py-cpuinfo
transformers >= 4.51.1
huggingface-hub[hf_xet] >= 0.33.0  # Required for Xet downloads.
tokenizers >= 0.21.1  # Required for fast incremental detokenization.
protobuf # Required by LlamaTokenizer.
fastapi[standard] >= 0.115.0 # Required by FastAPI's form models in the OpenAI API server's audio transcriptions endpoint.
aiohttp
openai >= 1.87.0, <= 1.90.0 # Ensure modern openai package (ensure ResponsePrompt exists in type.responses and max_completion_tokens field support)
pydantic >= 2.10
prometheus_client >= 0.18.0
pillow  # Required for image processing
prometheus-fastapi-instrumentator >= 7.0.0
tiktoken >= 0.6.0  # Required for DBRX tokenizer
lm-format-enforcer >= 0.10.11, < 0.11
llguidance >= 0.7.11, < 0.8.0; platform_machine == "x86_64" or platform_machine == "arm64" or platform_machine == "aarch64"
outlines_core == 0.2.10
# required for outlines backend disk cache
diskcache == 5.6.3
lark == 1.2.2
xgrammar == 0.1.19; platform_machine == "x86_64" or platform_machine == "aarch64" or platform_machine == "arm64"
typing_extensions >= 4.10
filelock >= 3.16.1 # need to contain https://github.com/tox-dev/filelock/pull/317
partial-json-parser # used for parsing partial JSON outputs
pyzmq >= 25.0.0
msgspec
gguf >= 0.13.0
importlib_metadata; python_version < '3.10'
mistral_common[opencv] >= 1.6.2
opencv-python-headless >= 4.11.0    # required for video IO
pyyaml
six>=1.16.0; python_version > '3.11' # transitive dependency of pandas that needs to be the latest version for python 3.12
setuptools>=77.0.3,<80; python_version > '3.11' # Setuptools is used by triton, we need to ensure a modern version is installed for 3.12+ so that it does not try to import distutils, which was removed in 3.12
einops # Required for Qwen2-VL.
compressed-tensors == 0.10.2 # required for compressed-tensors
depyf==0.18.0 # required for profiling and debugging with compilation config
cloudpickle # allows pickling lambda functions in model_executor/models/registry.py
watchfiles # required for http server to monitor the updates of TLS files
python-json-logger # Used by logging as per examples/others/logging_configuration.md
scipy # Required for phi-4-multimodal-instruct
ninja # Required for xgrammar, rocm, tpu, xpu
pybase64 # fast base64 implementation

numba == 0.60.0; python_version == '3.9' # v0.61 doesn't support Python 3.9. Required for N-gram speculative decoding
numba == 0.61.2; python_version > '3.9'

# Dependencies for CPUs
packaging>=24.2
setuptools>=77.0.3,<80.0.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.6.0+cpu; platform_machine == "x86_64" # torch>2.6.0+cpu has performance regression on x86 platform, see https://github.com/pytorch/pytorch/pull/151218
torch==2.7.0; platform_system == "Darwin"
torch==2.7.0; platform_machine == "ppc64le" or platform_machine == "aarch64"

# required for the image processor of minicpm-o-2_6, this must be updated alongside torch
torchaudio; platform_machine != "ppc64le" and platform_machine != "s390x"
torchaudio==2.7.0; platform_machine == "ppc64le"

# required for the image processor of phi3v, this must be updated alongside torch
torchvision; platform_machine != "ppc64le" and platform_machine != "s390x"
torchvision==0.22.0; platform_machine == "ppc64le"
datasets # for benchmark scripts

# Intel Extension for PyTorch, only for x86_64 CPUs
intel-openmp==2024.2.1; platform_machine == "x86_64"
intel_extension_for_pytorch==2.6.0; platform_machine == "x86_64" # torch>2.6.0+cpu has performance regression on x86 platform, see https://github.com/pytorch/pytorch/pull/151218
py-libnuma; platform_system != "Darwin"
psutil; platform_system != "Darwin"
triton==3.2.0; platform_machine == "x86_64" # Triton is required for torch 2.6+cpu, as it is imported in torch.compile.
		'';
	};
    };
  };

  packages = [
    pkgs.zsh
    pkgs.git
    pkgs.curl
    pkgs.wget
    pkgs.gcc12
    pkgs.ollama
  ] ++ lib.optionals (pkgs.stdenv.isx86_64) [
    pkgs.numactl
  ];
  
  enterShell = ''
    git --version

    uv venv --python 3.12 --seed;
    source .devenv/state/venv/bin/activate;
  '' + (if pkgs.stdenv.isDarwin then ''
  pip install -e .
'' else "");

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  processes = {
      open-webui.exec = "uvx --python 3.11 open-webui@latest serve";
      vllm-serve-qwen.exec = "vllm serve qwen/Qwen1.5-0.5B-Chat";
      ollama-serve.exec = "ollama serve";
  };

  git-hooks.hooks = {
    nixpkgs-fmt.enable = true;
  };

  env = {
    DATA_DIR = ".open-webui";
  };
}
