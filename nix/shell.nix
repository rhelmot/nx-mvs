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
