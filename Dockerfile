FROM python:3.13.15-slim-trixie@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b AS whisper-build
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp /src \
    && test "$(git -C /src rev-parse HEAD)" = d1be6fde11ac6e0407606b4e42fe72d34add8037 \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
       -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
       -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /src/build --target whisper-cli -j 1

FROM python:3.13.15-slim-trixie@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b AS llama-download
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 3 -o /tmp/llama.tar.gz https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10777-e3de715/llama-prism-b10777-e3de715-bin-ubuntu-x64.tar.gz \
    && echo '3f97b6fbec0065fede2be92f05df531c1050636fc2613e1ba0d975249df19713  /tmp/llama.tar.gz' | sha256sum -c - \
    && mkdir /llama && tar xzf /tmp/llama.tar.gz -C /llama --strip-components=1

FROM python:3.13.15-slim-trixie@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 libcurl4t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home voicey
WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY --from=whisper-build /src/build/bin/whisper-cli /opt/runtimes/whisper/bin/whisper-cli
COPY --from=llama-download /llama/ /opt/runtimes/llama/
COPY voicey /app/voicey
COPY main.py /app/main.py
COPY config /app/config
ENV DATA_DIR=/data PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
USER 10001:10001
CMD ["python", "main.py"]
