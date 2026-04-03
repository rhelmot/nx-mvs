(import <nixpkgs> {}).callPackage ({
    mkShell,
    python3,
    pyright,
    cmake,
    ninja,
}:
mkShell ({
    nativeBuildInputs = [
        cmake
        ninja
        pyright
        (python3.withPackages (p: with p; [
            networkx
            nanobind
            scikit-build-core
            wheel
            build
            setuptools
        ]))
    ];
})) {}
