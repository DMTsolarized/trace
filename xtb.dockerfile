# Use Ubuntu 22.04
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential gfortran cmake git wget curl \
    libopenblas-dev liblapack-dev libomp-dev \
    python3 python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=4

# Install CREST
WORKDIR /opt
RUN git clone https://github.com/grimme-lab/crest.git

# Directory for persistent xTB build
VOLUME ["/opt/xtb_build"]

# Build xTB if not already present
WORKDIR /opt/xtb_build
RUN if [ ! -f build/bin/xtb ]; then \
    git clone https://github.com/grimme-lab/xtb.git /opt/xtb_src && \
    mkdir -p build && cd build && \
    cmake /opt/xtb_src \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_FLAGS="-O3 -fopenmp" && \
    make -j$(nproc); \
    fi

# Create wrapper to force ASCII output
RUN mkdir -p /opt/xtb_build/bin
RUN cat <<'EOF' > /opt/xtb_build/bin/xtb_wrapper
#!/bin/bash
export LANG=C
export LC_ALL=C
export FORTRAN_BINARY_MODE=ascii
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec /opt/xtb_build/build/bin/xtb "$@"
EOF

RUN chmod +x /opt/xtb_build/bin/xtb_wrapper

# Add xTB wrapper and CREST to PATH
ENV PATH="/opt/xtb_build/bin:/opt/crest:$PATH"

# Set working directory
WORKDIR /workdir
VOLUME ["/workdir"]

# Default shell
CMD ["/bin/bash"]
