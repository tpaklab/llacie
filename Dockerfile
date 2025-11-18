FROM python:3.11-slim-bookworm

# Some steps are specific to the linux/arm64 platform
ARG TARGETPLATFORM

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including PostgreSQL client libraries and build tools
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    # PostgreSQL client libraries
    libpq-dev \
    postgresql-client \
    # Build dependencies for psycopg2 and other packages
    gcc-12 \
    g++-12 \
    # Git and make for potential repository dependencies
    git \
    make \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the application
RUN useradd -m -u 1000 -s /bin/bash llacie

# Set gcc-12 as default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Install python build tools as root
RUN pip install --upgrade pip setuptools wheel flit

# Setting up llacie itself
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY --chown=llacie:llacie pyproject.toml README.md ./

# Copy the entire application code
COPY --chown=llacie:llacie . .

# Switch to non-root user
USER llacie

# Add user's local bin to PATH
ENV PATH="/home/llacie/.local/bin:${PATH}"

# Ensure certain cache directories exist, then switch back to the llacie directory
WORKDIR /home/llacie/.cache
WORKDIR /home/llacie/.cache/outlines
WORKDIR /app

# Install the llacie package in editable mode (this installs all dependencies from pyproject.toml)
# For TARGETPLATFORM=linux/arm64, use abetlen's prebuilt llama-cpp-python wheel 
#    as it doesn't compile correctly from source
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip install --user -e . --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; \
    else \
        pip install --user -e .; \
    fi

# Set the default command to show help
CMD ["llacie", "--help"]
