EXPORT DATASETS_ROOT="datasets"

# Archive omniscripts for the upload 
tar -cf omniscripts.tar  --exclude=omniscripts.tar ../../.

# Build the image, use optional `--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}` to configure proxy.
docker build -t modin-project/benchmarks-reproduce:latest -f ./Dockerfile .

# Download data
./load_data.sh

# Create folder for the experiment results
mkdir results

# Run experiments
