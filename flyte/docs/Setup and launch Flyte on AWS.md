# Setup Flyte on AWS

## OS: Ubuntu 20.04

## Installation

1. Install Flytekit

   ```
   pip3 install flytekit --upgrade
   ```

   proven to work for version:

   ```
   0.26.0version.BuildInfo{Version:"v3.8.0", GitCommit:"d14138609b01886f544b2025f5000351c9eb092e", GitTreeState:"clean", GoVersion:"go1.17.5"}
   ```

2. Install FlyteCTL

   ```
   curl -sL https://ctl.flyte.org/install | bash
   ```

   proven to work for version:

   ```
   {
     "App": "flytectl",
     "Build": "c726223",
     "Version": "0.4.19",
     "BuildTime": "2022-02-07 09:00:16.98933728 +0000 UTC m=+0.016159657"
   }
   ```

3. Install Docker

   ```
   sudo apt-get update
   
   sudo apt-get install \
       ca-certificates \
       curl \
       gnupg \
       lsb-release
   
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   
   sudo apt-get update
   
   apt-cache madison docker-ce
   
   sudo apt-get install docker-ce=<VERSION_STRING> docker-ce-cli=<VERSION_STRING> containerd.io
   ```

   Post-installation steps:

   ```
   sudo groupadd docker
   
   sudo usermod -aG docker $USER
   
   newgrp docker
   
   sudo systemctl enable docker.service
   sudo systemctl enable containerd.service
   ```

   proven to work for version:

   ```
   Client: Docker Engine - Community
    Version:           20.10.12
    API version:       1.41
    Go version:        go1.16.12
    Git commit:        e91ed57
    Built:             Mon Dec 13 11:45:33 2021
    OS/Arch:           linux/amd64
    Context:           default
    Experimental:      true
   
   Server: Docker Engine - Community
    Engine:
     Version:          20.10.12
     API version:      1.41 (minimum version 1.12)
     Go version:       go1.16.12
     Git commit:       459d0df
     Built:            Mon Dec 13 11:43:42 2021
     OS/Arch:          linux/amd64
     Experimental:     false
    containerd:
     Version:          1.4.12
     GitCommit:        7b11cfaabd73bb80907dd23182b9347b4245eb5d
    runc:
     Version:          1.0.2
     GitCommit:        v1.0.2-0-g52b36a2
    docker-init:
     Version:          0.19.0
     GitCommit:        de40ad0
   ```

4. Login to Docker

   ```
   docker login
   ```

5. Install kubectl

   ```
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   ```

   To install specific version, for example, v1.21.0:

   ```
   curl -LO https://dl.k8s.io/release/v1.21.0/bin/linux/amd64/kubectl
   ```

   

   ```
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   ```

   proven to work for version:

   ```
   Client Version: version.Info{Major:"1", Minor:"21", GitVersion:"v1.21.1", GitCommit:"5e58841cce77d4bc13713ad2b91fa0d961e69192", GitTreeState:"clean", BuildDate:"2021-05-12T14:18:45Z", GoVersion:"go1.16.4", Compiler:"gc", Platform:"linux/amd64"}
   ```

6. Install AWS CLI

   Install unzip

   ```
   sudo apt install unzip
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   ```

   proven to work for version:

   ```
   aws-cli/2.4.14 Python/3.8.8 Linux/5.11.0-1027-aws exe/x86_64.ubuntu.20 prompt/off
   ```

7. Configure AWS CLI

   ```
   aws configure
   ```

8. Install Terraform

   ```
   sudo apt-get update && sudo apt-get install -y gnupg software-properties-common curl
   curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
   sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
   sudo apt-get update && sudo apt-get install terraform
   ```

9. Install Opta

   To install latest vesion:

   ```
   /bin/bash -c "$(curl -fsSL https://docs.opta.dev/install.sh)"
   ```

   To install specific version:

   ```
   VERSION=0.x /bin/bash -c "$(curl -fsSL https://docs.opta.dev/install.sh)"
   ```

   proven to work for version:

   ```
   v0.24.3
   ```

   Symlink the opta binary to one of your path directories

   ```
   sudo ln -fs ~/.opta/opta /usr/local/bin/opta
   ```

10. Install Helm

   ```
   curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
   chmod 700 get_helm.sh
   ./get_helm.sh
   ```

   proven to work for version:

   ```
   version.BuildInfo{Version:"v3.8.0", GitCommit:"d14138609b01886f544b2025f5000351c9eb092e", GitTreeState:"clean", GoVersion:"go1.17.5"}
   ```

11. Initialize project

    ```
    pyflyte init myflyteapp
    ```

11. Clone Flyte repository

    ```
    git clone https://github.com/flyteorg/flyte.git
    ```

    proven to work for commit `94327d6e9f29c3034e714577a9df27f6958ef170` 

## Configuration

1. Disable changing public access blocks for s3 buckets

   1. Edit file `~/.opta/modules/aws_s3/aws-s3.yaml`:

      ```
        - name: block_public
          user_facing: true
          validator: bool(required=False)
          description: Block all public access.
          default: true
      ```

      to:

      ```
        - name: block_public
          user_facing: true
          validator: bool(required=False)
          description: Block all public access.
          default: false
      ```

   2. Edit file `~/.opta/modules/aws_s3/tf_module/replication.tf`:

      Comment out these lines:

      ```
      #resource "aws_s3_bucket_public_access_block" "block_for_replica" {
      #  count  = var.same_region_replication ? 1 : 0
      #  bucket = aws_s3_bucket.replica[0].id
      #
      #  block_public_acls       = true
      #  block_public_policy     = true
      #  ignore_public_acls      = true
      #  restrict_public_buckets = true
      #}
      ```

      and these lines in `resource "aws_s3_bucket_policy" "replica_bucket_policy"`:

      ```
      #  depends_on = [
      #    aws_s3_bucket_public_access_block.block[0]
      #  ]
      ```

   3. Edit `~/.opta/modules/aws_base/tf_module/log_bucket.tf`

      Comment out these lines:

      ```
      #resource "aws_s3_bucket_public_access_block" "block" {
      #  bucket = aws_s3_bucket.log_bucket.id
      
      #  block_public_acls       = false
      #  block_public_policy     = false
      #  ignore_public_acls      = false
      #  restrict_public_buckets = false
      #}
      
      ```

      and these lines in `resource "aws_s3_bucket_policy" "log_bucket_policy"`:

      ```
      #  depends_on = [
      #    aws_s3_bucket_public_access_block.block
      #  ]
      ```

2. Edit `flyte/opta/aws/env.yaml`:

   1. Replace values:

      * <account_id>: your AWS account ID
      * <region>: your AWS region
      * <env_name>: a name for the new isolated cloud environment which is going to be created (e.g., test-name)
      * <your_company>: your company or organization’s name (e.g., test-org-name)

   2. Comment out lines:

      ```
      #  - type: dns
      #    domain: <domain>
      #    delegated: false # set to true once ready https://docs.opta.dev/miscellaneous/ingress/
      ```

   3. Change default k8s version and instance type

      ```
        - type: k8s-cluster
          max_nodes: 15
          k8s_version: "1.21"
          node_instance_type: "r5.large"
      ```

      \* You can change node_instance_type to suit your needs 

3. Edit `flyte/opta/aws/flyte.yaml`:

   1. Replace values:

      * <account_id>: your AWS account ID
      * <region>: your AWS region

   2. Change default chart version:

      ```
        - type: helm-chart
          chart: "../../charts/flyte-core" # NOTE: relative path to chart
          namespace: flyte
          timeout: 600
          create_namespace: true
          values_file: "../../charts/flyte-core/values-eks.yaml" # NOTE: relative path to values yaml
          chart_version: "v0.19.1"
      ```

   3. (Optional) Change default values for task resource limits

      paste these lines under `values:`

      ```
            configmap:
              task_resource_defaults:
                task_resources:
                  limits:
                    memory: 20Gi
      ```

      \* Default limit for memory is `1Gi`. 

   4. (Optional) Change default values for task resource limits

      ```
      cluster_resource_manager:
              enabled: true
              config:
                cluster_resources:
                  customData:
                    - production:
                        - defaultIamRole:
                            value: "${{module.userflyterole.role_arn}}"
                        - projectQuotaCpu:
                            value: "6"
                        - projectQuotaMemory:
                            value: "6000Mi"
                    - staging:
                        - defaultIamRole:
                            value: "${{module.userflyterole.role_arn}}"
                        - projectQuotaCpu:
                            value: "6"
                        - projectQuotaMemory:
                            value: "6000Mi"
                    - development:
                        - defaultIamRole:
                            value: "${{module.userflyterole.role_arn}}"
                        - projectQuotaCpu:
                            value: "6"
                        - projectQuotaMemory:
                            value: "6000Mi"
      ```

      `projectQuotaCpu`and `projectQuotaMemory` can be changed to suit your needs

      For example:

      ```
                    - development:
                        - defaultIamRole:
                            value: "${{module.userflyterole.role_arn}}"
                        - projectQuotaCpu:
                            value: "32"
                        - projectQuotaMemory:
                            value: "64Gi"
      ```

## Deployment

1. Deploy kubernetes cluster

   ```
   cd flyte/opta/aws
   opta apply -c env.yaml --auto-approve
   ```

2. Deploy Flyte on a cluster

   ```
   opta apply -c flyte.yaml --auto-approve
   ```

3. Update kubeconfig

   ```
   aws eks --region us-west-2 update-kubeconfig --name opta-test-name
   ```

4. Create or update Flyte config

   ```
   kubectl get service  -n flyte | grep flyteadmin
   ```

   Copy address.

   If you do not have `~/.flyte/config.yaml` file:

   ```
   flytectl config init --host=<FLYTEADMIN_URL>:81 --storage --insecure
   ```

   `<FLYTEADMIN_URL>` is the address from previous command

   If you already have `~/.flyte/config.yaml` file, edit it:

   ```
     endpoint: dns:///<FLYTEADMIN_URL>:81
   ```

   `~/.flyte/config.yaml` should look like this:

   ```
   admin:
     # For GRPC endpoints you might want to use dns:///flyte.myexample.com
     endpoint: dns:///<FLYTEADMIN_URL>:81
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
         region: <REGION> # Example: us-east-2
     container: <BUCKET> # Example my-bucket. Flyte k8s cluster / service account for execution should have read access to this bucket
   ```

   `<REGION>` and `<BUCKET>` should be replaced with your aws region of choice and a bucket, created by opta. Bucket name is currently `<env-name>-service-flyte`, where `<env_name>` is a value of `name: ` in `env.yaml`.

   \* Check that value of `endpoint:` has `dns:///` and not `dns://`.

5. Get access for Flyte Dashboard

   ```
   kubectl get ingress -n flyte
   ```

   Paste the link to the browser to get access to Dashboard

## Workflow execution

1. Build and push docker container

   ```
   cd myflyteapp
   docker build . --tag <registry/repo:version>
   docker push <registry/repo:version>
   ```

   `registry` is your DockerHub login

   For example:

   ```
   docker build . --tag myname/flyte-test:001
   docker push myname/flyte-test:001
   ```

2. Package the workflow

   ```
   pyflyte --pkgs flyte.workflows package -f --image <registry/repo:version>
   ```

3. Upload this package to the Flyte backend

   ```
   flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v1
   ```

   `--project flytesnacks` and `--domain development` can be changed to other existing projects and domains.

   After this command, you will be able to launch your workflow with Flyte Dashboard. 

## Cluster Destruction

```
cd flyte/opta/aws
opta destroy -c flyte.yaml --auto-approve
opta destroy -c env.yaml --auto-approve
```
