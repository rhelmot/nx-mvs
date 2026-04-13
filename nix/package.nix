{
    lib,
    buildPythonPackage,
    cmake,
    ninja,
    nlohmann_json,
    scikit-build-core,
    nanobind,
    networkx,
}:
buildPythonPackage {
    pname = "nx-mvs";
    version = "0.1.0";
    src = ./..;
    pyproject = true;
    dontConfigure = true;

    build-system = [ scikit-build-core nanobind ];
    dependencies = [ networkx ];
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
