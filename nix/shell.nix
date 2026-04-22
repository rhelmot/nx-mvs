{
    mkShell,
    python3,
    pyright,
    perf,
    strace,
    py-spy,
    procps,
    gdb,
    graphviz,
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
            xdot
            pygraphviz
            matplotlib
        ]))
    ];
    nativeBuildInputs = [
        pyright
        perf
        strace
        py-spy
        procps
        gdb
        graphviz
    ];

    shellHook = ''
      export PYTHONPATH="$PYTHONPATH:$PWD/src"
    '';
}
