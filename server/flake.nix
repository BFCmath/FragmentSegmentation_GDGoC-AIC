{
  description = "Python flake for running the backend";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {
    devShell.${system} = pkgs.mkShell {
      buildInputs = [
        (pkgs.python312.withPackages (python-pkgs: [
          # User libraries
          python-pkgs.uvicorn
          python-pkgs.fastapi
          python-pkgs.python-multipart
          
          # Pytorch
          python-pkgs.torch
          python-pkgs.torchvision
        ]))

        pkgs.pyright
      ];
    };
  };
}
