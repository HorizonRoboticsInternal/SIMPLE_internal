# syntax=docker/dockerfile:1.7

ARG SIMPLE_BASE_IMAGE=simple-holobrain-simple-base:latest
FROM ${SIMPLE_BASE_IMAGE}

USER root
WORKDIR /workspace/robotics

ARG HOLOBRAIN_PYTHON_VERSION=3.11.15
ARG TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV HOLOBRAIN_ENV=/opt/envs/holobrain
ENV HOLOBRAIN_PYTHON=/opt/envs/holobrain/bin/python
ENV PIP_NO_CACHE_DIR=1
ENV UV_HTTP_TIMEOUT=1800
ENV UV_HTTP_RETRIES=10

# Keep the two runtimes isolated. SIMPLE remains in /workspace/SIMPLE/.venv;
# this environment contains only HoloBrain model/server dependencies.
RUN uv python install ${HOLOBRAIN_PYTHON_VERSION} && \
    uv venv --python ${HOLOBRAIN_PYTHON_VERSION} --seed ${HOLOBRAIN_ENV}

COPY SIMPLE/docker/simple_holobrain/holobrain-requirements.txt /tmp/holobrain-requirements.txt

# Install torch first because FlashAttention and PyTorch3D compile against it.
RUN ${HOLOBRAIN_PYTHON} -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchvision==0.22.0 && \
    ${HOLOBRAIN_PYTHON} -m pip install \
        -r /tmp/holobrain-requirements.txt

# Only copy runtime source. Large jobs, datasets, checkpoints, existing venvs,
# and git metadata are excluded by Dockerfile.dockerignore.
COPY Psi0/src /workspace/robotics/Psi0/src
COPY Psi0/scripts /workspace/robotics/Psi0/scripts
COPY robo_orchard_lab/robo_orchard_lab /workspace/robotics/robo_orchard_lab/robo_orchard_lab
COPY robo_orchard_lab/scripts /workspace/robotics/robo_orchard_lab/scripts
COPY robo_orchard_lab/projects/__init__.py /workspace/robotics/robo_orchard_lab/projects/__init__.py
COPY robo_orchard_lab/projects/holobrain_internal/common /workspace/robotics/robo_orchard_lab/projects/holobrain_internal/common
COPY robo_orchard_lab/projects/holobrain_internal/lightwheel /workspace/robotics/robo_orchard_lab/projects/holobrain_internal/lightwheel
COPY lw_benchhub/third_party/pytorch3d /workspace/robotics/lw_benchhub/third_party/pytorch3d

# robo_orchard_core is small, but the full robo_orchard_lab dependency set
# includes training-only packages. Install core without pulling opencv-python
# (the headless variant is already pinned above), and import lab from source.
RUN ${HOLOBRAIN_PYTHON} -m pip install --no-deps \
        "robo_orchard_core@git+https://github.com/HorizonRobotics/robo_orchard_core.git@98dacf63ff6fa0c7e5684af81e9a1b93d106bd74" && \
    MAX_JOBS=8 ${HOLOBRAIN_PYTHON} -m pip install --no-build-isolation \
        flash-attn==2.7.4.post1 && \
    FORCE_CUDA=1 MAX_JOBS=8 ${HOLOBRAIN_PYTHON} -m pip install \
        --no-build-isolation --no-deps \
        -e /workspace/robotics/lw_benchhub/third_party/pytorch3d

COPY robo_orchard_lab/holomotion_tokenizer /workspace/robotics/robo_orchard_lab/holomotion_tokenizer

ENV PYTHONPATH=/workspace/SIMPLE/src:/workspace/robotics/Psi0/src:/workspace/robotics/lw_benchhub:/workspace/robotics/lw_benchhub/third_party/pytorch3d
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

COPY SIMPLE/src /workspace/SIMPLE/src
COPY SIMPLE/docker/simple_holobrain/entrypoint.sh /usr/local/bin/simple-holobrain
COPY SIMPLE/docker/simple_holobrain/smoke_test.py /workspace/SIMPLE/docker/simple_holobrain/smoke_test.py
RUN chmod 755 /usr/local/bin/simple-holobrain && \
    mkdir -p /models /eval-data /outputs && \
    PYTHONPATH=/workspace/robotics/robo_orchard_lab:${PYTHONPATH} \
        ${HOLOBRAIN_PYTHON} /workspace/SIMPLE/docker/simple_holobrain/smoke_test.py --imports-only && \
    rm -rf /workspace/robotics/robo_orchard_lab \
           /workspace/robotics/deploy

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/simple-holobrain"]
CMD ["shell"]
