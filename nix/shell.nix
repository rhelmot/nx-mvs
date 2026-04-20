{
    mkShell,
    python3,
    pyright,
    perf,
    strace,
    py-spy,
}:
mkShell {
    buildInputs = [
        (python3.withPackages (p: with p; [
            networkx
            pytest
            pydot
            build
            scikit-build-core
            nanobind
            cmake
            ninja
        ]))
    ];
    nativeBuildInputs = [
        pyright
        perf
        strace
        py-spy
    ];

    shellHook = ''
      export PYTHONPATH="$PYTHONPATH:$PWD/src"
    '';
}
