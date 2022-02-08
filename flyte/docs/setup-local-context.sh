#!/bin/bash -e

# Create EC2 instance, for example - c5n.2xlarge

# Prerequisits
# - Docker

# WEB view: https://github.com/k3s-io/k3s/releases/tag/v1.21.1%2Bk3s1
wget https://github.com/k3s-io/k3s/releases/download/v1.21.1%2Bk3s1/k3s
sudo mv k3s /usr/local/bin/
chmod +x /usr/local/bin/k3s
KUBERNETES_API_PORT=6443
sudo k3s server --docker --no-deploy=traefik --no-deploy=servicelb --no-deploy=local-storage --no-deploy=metrics-server --https-listen-port=${KUBERNETES_API_PORT} &> /home/ubuntu/log/k3s.log &
sleep 10
# check health
sudo k3s kubectl get namespace

sudo k3s kubectl create -f https://raw.githubusercontent.com/flyteorg/flyte/master/deployment/sandbox/flyte_generated.yaml

# This step seems to be the key to context switching.
flytectl config init --host='localhost:30081' --insecure
# TODO: add "/" in config.yml for endpoint field - Flyte bug

# Prepare workload to execute on k8s cluster
pyflyte init myflyteapp
# TODO: Build and push docker container - need to add the steps
# some commands
pyflyte --pkgs flyte.workflows package -f --image <registry/repo:version>

flytectl register files --project flytesnacks --domain development --archive flyte_example/flyte-package.tgz --version v1

# TODO: port-worwarding: 30081:localhost:30081 in active PuTTY session
# you are ready to launch workloads via Flyte Console

# These steps can be used without WEB interface
flytectl get launchplan --project flytesnacks --domain development myapp.workflows.example.my_wf --latest --execFile exec_spec.yaml
flytectl create execution --project flytesnacks --domain development --execFile exec_spec.yaml
flytectl get execution --project flytesnacks --domain development <exec name>

# check all workflows
flytectl get workflow -p flytesnacks -d development

sudo k3s kubectl delete  -f https://raw.githubusercontent.com/flyteorg/flyte/master/deployment/sandbox/flyte_generated.yaml
sudo k3s kubectl delete namespace flyteexamples-development flyteexamples-staging flyteexamples-production  flytesnacks-development flytesnacks-staging flytesnacks-production

# no need to do this - just forward in PuTTY - 30081:localhost:30081
# sudo k3s kubectl port-forward -n flyte service/flyteconsole 8080:80

# need to update memory quotas
# bugs when setuping limits - in configs ok, but in pods used only requests - report to Flyte
# For running taxi - change task_resource_defaults, cluster_resources
