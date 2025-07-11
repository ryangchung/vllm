{ pkgs, lib, config, inputs, ... }:

{
  languages = {
    shell.enable = true;
    nix.enable = true;
    python = {
      enable = true;
	    uv.enable = true;
	    venv.enable = true;
    };
  };

  packages = [
    pkgs.zsh
    pkgs.git
    pkgs.curl
    pkgs.wget
    pkgs.gcc12
    pkgs.numactl
  ];
  enterShell = ''
    git --version

    uv venv --python 3.12 --seed;
    source .devenv/state/venv/bin/activate;

    # uvx --python 3.11 open-webui@latest serve
  '';

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  git-hooks.hooks = {
    nixpkgs-fmt.enable = true;
  };


  env = {
    DATA_DIR = ".open-webui";
  };
}
