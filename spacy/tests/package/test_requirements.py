import re
from pathlib import Path


def test_build_dependencies():
    # Check that library requirements are pinned exactly the same across different setup files.
    libs_ignore_requirements = [
        "cython",
        "pytest",
        "pytest-timeout",
        "mock",
        "flake8",
        "hypothesis",
        "pre-commit",
        "cython-lint",
        "black",
        "isort",
        "mypy",
        "types-dataclasses",
        "types-mock",
        "types-requests",
        "types-setuptools",
    ]
    # ignore language-specific packages that shouldn't be installed by all
    libs_ignore_setup = [
        "numpy",
        "fugashi",
        "mecab-ko",
        "pythainlp",
        "sudachipy",
        "sudachidict_core",
        "spacy-pkuseg",
        "thinc-apple-ops",
    ]

    # check requirements.txt
    req_dict = {}

    root_dir = Path(__file__).parent
    req_file = root_dir / "requirements.txt"
    with req_file.open() as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line.startswith("#"):
                lib, v = _parse_req(line)
                if lib and lib not in libs_ignore_requirements:
                    req_dict[lib] = v
    # check setup.cfg and compare to requirements.txt
    # also fails when there are missing or additional libs
    setup_file = root_dir / "setup.cfg"
    with setup_file.open() as f:
        lines = f.readlines()

    setup_keys = set()
    for line in lines:
        line = line.strip()
        if not line.startswith("#"):
            lib, v = _parse_req(line)
            if lib and not lib.startswith("cupy") and lib not in libs_ignore_setup:
                req_v = req_dict.get(lib, None)
                assert (
                    req_v is not None
                ), "{} in setup.cfg but not in requirements.txt".format(lib)
                assert (lib + v) == (lib + req_v), (
                    "{} has different version in setup.cfg and in requirements.txt: "
                    "{} and {} respectively".format(lib, v, req_v)
                )
                setup_keys.add(lib)

    # check pyproject.toml and compare the versions of the libs to requirements.txt
    # does not fail when there are missing or additional libs
    toml_file = root_dir / "pyproject.toml"
    with toml_file.open() as f:
        lines = f.readlines()
    pyproject_keys = set()
    for line in lines:
        line = line.strip().strip(",").strip('"')
        if not line.startswith("#"):
            lib, v = _parse_req(line)
            if lib and lib not in libs_ignore_requirements:
                pyproject_keys.add(lib)
                req_v = req_dict.get(lib, None)
                assert (lib + v) == (lib + req_v), (
                    "{} has different version in pyproject.toml and in requirements.txt: "
                    "{} and {} respectively".format(lib, v, req_v)
                )

    # if fail: requirements.txt contains a lib not in setup.cfg or pyproject.toml
    assert set(setup_keys).union(set(pyproject_keys)) == set(req_dict.keys())


def _parse_req(line):
    lib = re.match(r"^[a-z0-9\-]*", line).group(0)
    v = line.replace(lib, "").strip()
    if not re.match(r"^[<>=][<>=].*", v):
        return None, None
    return lib, v
