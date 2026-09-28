# HE-LRM: Encrypted Deep Learning Recommendation Models under FHE

Anonymous artifact for the paper *HE-LRM: Encrypted Deep Learning
Recommendation Models using Fully Homomorphic Encryption*.

HE-LRM runs a complete Criteo DLRM (bottom MLP, 26 embedding-table lookups,
concatenation, top MLP) on CKKS ciphertexts. The embedding lookup is a
plaintext-matrix / encrypted-vector product over deterministically
digit-decomposed, block-diagonally packed compressed tables, so it consumes one
multiplicative level and produces embeddings already packed for the rest of the
network. The repository is a fork of the public Orion FHE framework
(Ebel et al., ASPLOS 2025) extended with a GPU backend built on Cheddar
(Kim et al., 2024) and with an implementation of the IVE selector
(Cheon et al., 2026) on the same stack, so the two designs are timed on one
machine, one build, one clock.

## Layout

| Path | What it is |
|---|---|
| `orion/` | The Orion framework: `nn` layers, `core` compiler and packing, `models`, and the backends. |
| `orion/backend/lattigo/` | CPU backend (Go, Lattigo). Built by `tools/build_lattigo.py` at install time. |
| `orion/backend/cheddar/` | GPU backend added for this paper: a pybind11 extension over Cheddar, 64-bit and 32-bit word builds, the patches Cheddar needs, and its own [README](orion/backend/cheddar/README.md). |
| `examples/criteo/` | HE-LRM itself: dataset preparation, training with compressed embeddings, the FHE forward, the IVE selector, and every measurement script behind the paper's tables and figures. See its [README](examples/criteo/README.md). |
| `configs/`, `examples/run_*.py` | Orion's own examples (LoLA, MLP, ResNet-20) with CPU and GPU parameter files; useful to check a backend build. |
| `tests/` | Backend oracle tests (`--backend=lattigo|desilo|cheddar`) and unit tests. |

## Requirements

- Linux (tested on Ubuntu 22.04), Python 3.10 to 3.12, [uv](https://docs.astral.sh/uv/) or pip.
- Go 1.22 for the Lattigo CPU backend.
- For the GPU backend: an NVIDIA GPU, CUDA 12, CMake 3.24 or newer, gcc/g++ 11,
  and network access on first configure (Cheddar fetches RMM). The paper's GPU
  numbers come from an RTX PRO 6000 (96 GiB), an H100 (80 GiB), an A100
  (40 GiB) and an RTX 3090 (24 GiB); the CPU numbers from one thread of a
  24-core workstation with 256 GiB of RAM.

## Install

```bash
sudo apt update && sudo apt install -y build-essential git wget curl \
    ca-certificates python3 python3-pip python3-venv unzip pkg-config \
    libgmp-dev libssl-dev
# Go (Lattigo backend)
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
export PATH=/usr/local/go/bin:$PATH

git clone https://github.com/asplos2027-helrm/helrm.git helrm && cd helrm
uv sync            # or: pip install -e .
```

`uv sync` compiles the Lattigo backend. Check the install with an Orion
example on the CPU:

```bash
uv run examples/run_mlp.py
uv run --with pytest pytest tests/oracle/ -v -s
```

### GPU backend

Build Cheddar at the pinned upstream commit with the five patches, then the
extension, following [`orion/backend/cheddar/README.md`](orion/backend/cheddar/README.md).
Then:

```bash
uv run examples/run_mlp.py configs/mlp_cheddar.yml
uv run --with pytest pytest tests/oracle/ -v -s --backend=cheddar
```

## Reproducing the paper

Everything in the paper's Section 5 is produced by the scripts in
`examples/criteo/`; the [README there](examples/criteo/README.md) gives the
data and training steps, the environment every measured cell ran under, the
measurement protocol, and one command per table and figure.

## License

MIT, see [LICENSE](LICENSE). The framework this repository extends is
Orion, released under the same license; its notice is retained.
