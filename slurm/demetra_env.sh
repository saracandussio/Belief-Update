# Demetra environment. The repo stays in $HOME; everything heavy lives on the
# share: .venv, data/ and outputs/ are symlinks into $STORE, caches and temp
# files point there too. Sourced by the SLURM scripts; also run
# `source slurm/demetra_env.sh` in an interactive shell before working.

export SHARE="/share/ai-lab/scandussio"
export STORE="$SHARE/belief-update"
export PROJECT_PATH="/u/scandussio/Belief-Update"

# Python 3.11 from spack (the venv in $PROJECT_PATH/.venv was built with it)
case $- in *u*) _bu_u=1; set +u ;; esac
type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; }
module load python/3.11.7-gcc-13.2.0-b7gwkjx
[ -n "${_bu_u:-}" ] && set -u; unset _bu_u

# Caches and temp files
export HF_HOME="$SHARE/hf_cache"
export XDG_CACHE_HOME="$SHARE/cache"
export PIP_CACHE_DIR="$SHARE/cache/pip"
export CONDA_PKGS_DIRS="$SHARE/cache/conda_pkgs"
export VLLM_CACHE_ROOT="$SHARE/cache/vllm"
export TRITON_CACHE_DIR="$SHARE/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$SHARE/cache/torchinductor"
export TMPDIR="$SHARE/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
