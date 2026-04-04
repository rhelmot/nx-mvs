{
    lib,
    python3,
    cmake,
    ninja,
    nlohmann_json,
}:
python3.pkgs.buildPythonPackage {
    pname = "nx-mvs";
    version = "0.1.0";
    src = ./..;
    pyproject = true;
    dontConfigure = true;

    build-system = with python3.pkgs; [ scikit-build-core nanobind ];
    dependencies = with python3.pkgs; [ networkx ];
    nativeBuildInputs = [
        cmake
        ninja
    ];
    buildInputs = [
        nlohmann_json
    ];

    meta = {
        license = with lib.licenses; [ publicDomain gpl2Plus ];
    };
}
