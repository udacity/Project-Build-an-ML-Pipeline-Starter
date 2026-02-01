# ============================================================================
# Stage 1: Build conda environment
# ============================================================================
FROM condaforge/mambaforge:24.3.0-0 AS conda-builder

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean --all --yes

# ============================================================================
# Stage 2: Final development container
# ============================================================================
FROM mcr.microsoft.com/devcontainers/base:bookworm

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy conda from builder stage
COPY --from=conda-builder /opt/conda /opt/conda

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Configure conda and auto-activate nyc_airbnb_dev environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate nyc_airbnb_dev" >> /etc/bash.bashrc

# Configure for zsh (Oh My Zsh will be installed by devcontainer features)
RUN echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/zsh/zshrc && \
    echo 'conda activate nyc_airbnb_dev' >> /etc/zsh/zshrc

# Set working directory
WORKDIR /workspaces/Project-Build-an-ML-Pipeline-Starter

# Set default shell
SHELL ["/bin/bash", "-c"]
