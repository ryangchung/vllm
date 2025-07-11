{ pkgs, lib, config, inputs, ... }:
{
  languages = {
    nix.enable = true;
  };
  packages = [
    pkgs.zsh
    pkgs.git
  ];
  enterShell = ''
    git --version

    # uvx --python 3.11 open-webui@latest serve

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  git-hooks.hooks = {
    nixpkgs-fmt.enable = true;
  };


  env = {
    DATA_DIR = "~/.open-webui";
  };
}
