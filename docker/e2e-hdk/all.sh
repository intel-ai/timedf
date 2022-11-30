
# Create folder for the experiment
mkdir experiment

# Clone latest omniscripts  into the current dir &
# Archive omniscripts for the upload 
git clone https://github.com/intel-ai/omniscripts && \
    tar -cf omniscripts.tar  --exclude=omniscripts/docker/omniscripts.tar ../../.

# Build the image, use optional `--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}` to configure proxy.
docker build -t modin-project/benchmarks-reproduce:latest -f ./Dockerfile .
4. Dowload data: ./scripts/load_data.sh