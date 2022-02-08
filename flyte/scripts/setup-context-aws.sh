#!/bin/bash -e

# Prerequisits that should be installed manually
# curl
# if OS Ubuntu/Debian Linux: `language-pack-en` package installed: for OPTA
# export LC_CTYPE=en_US.UTF-8

################################### Deps installation #############################
# Installation low level dependecies
sudo apt-get install unzip git -y

# create new env with python=3.8
conda create -n flyte-aws python=3.8 -c conda-forge
conda activate flyte-aws

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

#### OPTA PREREQS INSTALLATION ####

# Install docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker ${USER}


# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(<kubectl.sha256)  kubectl" | sha256sum --check
chmod +x kubectl
mv ./kubectl $CONDA_PREFIX/bin
kubectl version --client

# AWS CLI (v2) (exist in conda forge only for v1)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip # (check integrity with gpg? https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#install-linux-verify)
./aws/install -i $CONDA_PREFIX/lib/aws_cli -b $CONDA_PREFIX/bin
aws --version
# aws configure  # interactive setup: Access Key, Secret Access Key, default region name, default output format
mkdir -p /home/ubuntu/.aws/
touch /home/ubuntu/.aws/config
cat <<EOF >/home/ubuntu/.aws/config
[default]
region = us-west-2
output = json
EOF

ACCESS_KEY=[AWS ACCESS KEY]
SECRET_ACCESS_KEY=[AWS SECRET ACCESS KEY]

touch /home/ubuntu/.aws/credentials
cat <<EOF >/home/ubuntu/.aws/credentials
[default]
aws_access_key_id = $ACCESS_KEY
aws_secret_access_key = $SECRET_ACCESS_KEY
EOF

# helm
conda install kubernetes-helm -c conda-forge -y

# Terraform

sudo apt-get update && sudo apt-get install -y gnupg software-properties-common curl
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

#### OPTA INSTALLATION ####
# we can't choose path
# curl -sL https://docs.opta.dev/install.sh | bash -s -- -b $CONDA_PREFIX/bin
/bin/bash -c "$(curl -fsSL https://docs.opta.dev/install.sh)"
echo 'export PATH="$PATH:/home/ubuntu/.opta"' >> ~/.bashrc && source ~/.bashrc
opta version  # v0.24.3

#### FLYTE INSTALLATION ####
# install flytectl
curl -sL https://ctl.flyte.org/install | bash -s -- -b $CONDA_PREFIX/bin

# install flytekit; comes with pandas
pip install flytekit --upgrade

################################### Setup configs #################################

## DEFAULT PARAMETERS - DEFINE YOUR VALUES
ENV_NAME=test-name
# 15 symbols max
YOUR_COMPANY=test-company
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
REGION=us-west-2

# update opta config for S3 buckets (not sure that this is needed - discussing with maintainers)
# Delete or comment resource "aws_s3_bucket_public_access_block" "block" and
# resource "aws_s3_bucket_policy" "log_bucket_policy" in ~/.opta/modules/aws_base/tf_module/log_bucket.tf
# resource "aws_s3_bucket_public_access_block" "block" in ~/.opta/modules/aws_s3/tf_module/bucket.tf
# resource "aws_s3_bucket_policy" "replica_bucket_policy" in ~/.opta/modules/aws_s3/tf_module/replication.tf
# + aws_s3_bucket_public_access_block

sed -i "34s/true/false/" ~/.opta/modules/aws_s3/aws-s3.yaml

sed -i '107,115 s/^/#/' ~/.opta/modules/aws_s3/tf_module/replication.tf
sed -i '152,154 s/^/#/' ~/.opta/modules/aws_s3/tf_module/replication.tf

sed -i '45,52 s/^/#/' ~/.opta/modules/aws_base/tf_module/log_bucket.tf
sed -i '132,134 s/^/#/' ~/.opta/modules/aws_base/tf_module/log_bucket.tf


git clone https://github.com/flyteorg/flyte.git

### EDIT flyte/opta/aws/env.yaml
# TODO: need to remove dns section (in auto way);
# Maybe TODO: update k8s-cluster; which default `node_instance_type`?
sed -i "s/<env_name>/$ENV_NAME/g" flyte/opta/aws/env.yaml
sed -i "s/<your_company>/$YOUR_COMPANY/g" flyte/opta/aws/env.yaml
sed -i "s/<account_id>/$ACCOUNT_ID/g" flyte/opta/aws/env.yaml
sed -i "s/<region>/$REGION/g" flyte/opta/aws/env.yaml
# For development: decrease max_nodes. How to choose number of nodes in auto way?
sed -i "s/15/4/g" flyte/opta/aws/env.yaml
sed -i '9,11 s/^/#/' flyte/opta/aws/env.yaml

# TEMP workaround: https://github.com/run-x/opta/issues/647
# add `k8s_version: 1.21` into flyte/opta/aws/env.yaml
# add `chart_version: "v0.19.1"` into flyte/opta/aws/flyte.yaml

# NOTE: Manual delete of dns section

### EDIT flyte/opta/aws/flyte.yaml
# Maybe TODO: `cluster_resources` do not need to be changed (for testing)
sed -i "s/<account_id>/$ACCOUNT_ID/g" flyte/opta/aws/flyte.yaml
sed -i "s/<region>/$REGION/g" flyte/opta/aws/flyte.yaml

### TODO: Edit Task default limits

###################################

cookiecutter https://github.com/flyteorg/flytekit-python-template.git --directory="simple-example" -c 005f8830448095a50e42c2e60e764d00fbed4eb8 --no-input
# added my workflows under ./flyte-example/myapp/workflows/
# for example - taxi_flyte.py

# changed cwd; s it really need to use DOCKERHUB?
DOCKER_HUB_REPO=rustamovazer/flyte-test
DOCKER_IMAGE_TAG=v1
DOCKER_USERNAME=[YOUR NAME]
DOCKER_PASSWORD=[YOUR PASSWORD]

docker build ./flyte_example --tag $DOCKER_HUB_REPO:$DOCKER_IMAGE_TAG
docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
docker push $DOCKER_HUB_REPO:$DOCKER_IMAGE_TAG

################################### Deploy ########################################

# Maybe failed first time, but ok in second launch. Why?
# With Flyte: commit 362a3d33c56f65c3584ecdc0da48c2f149c9b075 (HEAD, tag: v0.19.1) - nginx error
# try 94327d6e9f29c3034e714577a9df27f6958ef170 as Azer used
cd flyte/opta/aws/
opta apply -c env.yaml --auto-approve
opta apply -c flyte.yaml --auto-approve
cd ~

# Configures kubectl so that you can connect to an Amazon EKS cluster.
aws eks --region $REGION update-kubeconfig --name opta-$ENV_NAME
# AUTOMATE IT: get flyteadmin_url
kubectl get service -n flyte | grep flyteadmin
FLYTEADMIN_URL=$(kubectl get service  -n flyte | grep flyteadmin | awk  'NR==1{print $4}')
# TODO: update flyteadmin url
# flytectl config init --host=$FLYTEADMIN_URL:81 --storage --insecure
BUCKET= "$ENV_NAME-service-flyte"
rm ~/.flyte/config.yaml
cat <<EOT >> ~/.flyte/config.yaml
admin:
  # For GRPC endpoints you might want to use dns:///flyte.myexample.com
  endpoint: dns:///$FLYTEADMIN_URL:81
  authType: Pkce
  insecure: true
logger:
  show-source: true
  level: 0
storage:
  type: stow
  stow:
    kind: s3
    config:
      auth_type: iam
      region: $REGION # Example: us-east-2
  container: $BUCKET # Example my-bucket. Flyte k8s cluster / service account for execution should have read access to this bucket
EOT
# The need only for flytectl and it's default value
# export FLYTECTL_CONFIG=~/.flyte/config.yaml  # Where it comes from?

# Maybe TODO: edit ~/.flyte/config.yaml
# region: us-east-1
# container: <S3_BUCKET>,  where <S3_BUCKET> is the bucket in values-eks.yaml
# potential: sed -i "s/<S3_BUCKET>/$S3_BUCKET/g" ~/.flyte/config.yaml

# get address that should be paste in browser
kubectl get ingress -n flyte

################################### Launch ########################################

# package the workflow
cd flyte_example
pyflyte --pkgs myapp.workflows package -f --image $DOCKER_HUB_REPO:$DOCKER_IMAGE_TAG
cd ..

# changed x4 -> development; register - download image in cluster I suppose
# second launch possible without docker build stage? fast-register option?
# fast register option require extra setup step: https://docs.flyte.org/en/latest/getting_started/iterate.html#bonus-build-deploy-your-application-fast-er
flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v1 --logger.level=6

# TODO: launch the workflow in browser; need to think about CLI analogue
# CLI analogue: https://docs.flyte.org/en/latest/getting_started/iterate.html#bonus-build-deploy-your-application-fast-er
# 1. flytectl get launchplan --project flytesnacks --domain development flyte.workflows.example.my_wf --latest --execFile exec_spec.yaml
# 2. Modify the execution spec file and update the input params and save the file. Notice that the version would be changed to your latest one.
# 3. flytectl create execution --project flytesnacks --domain development --execFile exec_spec.yaml
# 4. flytectl get execution --project flytesnacks --domain development <execname>

################################### Destroy #######################################

cd flyte/opta/aws
opta destroy -c flyte.yml --auto-approve
opta destroy -c env.yml --auto-approve