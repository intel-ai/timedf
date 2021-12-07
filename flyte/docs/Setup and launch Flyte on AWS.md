# Setup Flyte on AWS

1. Install WSL2 Ubuntu

   In PowerShell as admin

   ```
   wsl --install
   ```

   then, restart your computer

   For the next steps use Ubuntu terminal

2. Install Terraform

   ```
   sudo apt-get update && sudo apt-get install -y gnupg software-properties-common curl
   curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
   sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
   sudo apt-get update && sudo apt-get install terraform
   ```

   Test installation

   ```
   terraform -help
   ```

3. Install flytekit

   ```
   curl -sL https://ctl.flyte.org/install | sudo bash -s -- -b /usr/local/bin
   ```

4. Install kubectl

   ```
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   kubectl version --client
   ```

5. Register on DockerHub

6. Install unzip package

   ```
   sudo apt install unzip
   ```

7. Install AWS CLI

   ```
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   ```

8. Configure AWS CLI

   ``` 
   aws configure
   ```

   ```
   AWS Access Key ID [None]: ****
   AWS Secret Access Key [None]: ****
   Default region name [None]: us-east-2
   Default output format [None]: json
   ```

9. Install Opta

   ```
   /bin/bash -c "$(curl -fsSL https://docs.opta.dev/install.sh)"
   ```

10. Add opta to PATH

    ```
    nano ~/.bashrc
    ```

       add line:

    ```
    export PATH="$HOME/.opta:~/.local/bin:$PATH"
    ```

       Save bashrc

       Restart Shell or use command

    ```
    source ~/.bashrc
    ```

11. Delete or comment `resource "aws_s3_bucket_public_access_block" "block"` and `resource "aws_s3_bucket_policy" "log_bucket_policy"` in `~/.opta/config/tf_modules/aws-base/log_bucket.tf`,

    `resource "aws_s3_bucket_public_access_block" "block"` in `~\.opta\config\tf_modules\aws-s3\bucket_tf`,

    `resource "aws_s3_bucket_policy" "replica_bucket_policy"` in `~\.opta\config\tf_modules\aws-s3\replication.tf`

    Do this if you don't have permissions to make private s3 buckets

12. Create Python virtual environment

    ```
    sudo apt update && sudo apt upgrade
    sudo apt install python3-pip
    sudo apt install python3.8-venv
    python3 -m venv ~/venv
    ```

13. Activate Python virtual environment

    ```
    source ~/venv/bin/activate
    ```

14. Install Flytekit

    ```
    pip install flytekit --upgrade
    ```

15. Clone Flyte template reository

    ```
    git clone https://github.com/flyteorg/flytekit-python-template.git myflyteapp
    ```

16. Copy `queries_on_flyte.py` to `~/myflyteapp/myapp/workflows/`

17. Clone Flyte repository

    ```
    git clone https://github.com/flyteorg/flyte.git
    ```

18. ```
    cd flyte/opta
    ```

19. Edit `env.yaml` and `flyte.yaml` in `flyte/opta`

    For `env.yaml`replace `<env_name>` with `flyte` and `<your_company>` with `orgname`,  `<account_id>` with  `980842202052`, `<region>` with  `us-east-2`, comment out or delete these lines:

    ```  
    type: dns
    domain: <domain>
    delegated: false # set to true once ready https://docs.opta.dev/miscellaneous/ingress/
    ```

    Also, change `type: k8s-cluster`

    ```
      - type: k8s-cluster
        max_nodes: 20
        min_nodes: 18
        node_instance_type: "r5.4xlarge"
    ```

    For `flyte.yaml` replace `<region>` with `us-east-2`, `<account_id>` with your AWS account ID.

    Also change `cluster_resources`:

    from:

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

    to:

    ```
    cluster_resource_manager:
            enabled: true
            config:
              cluster_resources:
                customData:
                  - x2:
                      - defaultIamRole:
                          value: "${{module.userflyterole.role_arn}}"
                      - projectQuotaCpu:
                          value: "32"
                      - projectQuotaMemory:
                          value: "256Gi"
                  - x4:
                      - defaultIamRole:
                          value: "${{module.userflyterole.role_arn}}"
                      - projectQuotaCpu:
                          value: "64"
                      - projectQuotaMemory:
                          value: "512Gi"
                  - x8:
                      - defaultIamRole:
                          value: "${{module.userflyterole.role_arn}}"
                      - projectQuotaCpu:
                          value: "128"
                      - projectQuotaMemory:
                          value: "1024Gi"
                  - x16:
                      - defaultIamRole:
                          value: "${{module.userflyterole.role_arn}}"
                      - projectQuotaCpu:
                          value: "256"
                      - projectQuotaMemory:
                          value: "2048Gi"
    ```

20. Create CloudWatch log group.

21. Edit `flyte\charts\flyte\values-eks.yaml`

    Change these lines

    ```
    userSettings:
      accountNumber: <ACCOUNT_NUMBER>
      accountRegion: <AWS_REGION>
      certificateArn: <CERTIFICATE_ARN>
      dbPassword: <DB_PASSWORD>
      rdsHost: <RDS_HOST>
      bucketName: <BUCKET_NAME>
      logGroup: <LOG_GROUP_NAME>
    ```

    You can safely delete lines:`certificateArn`, `dbPassword`, `rdsHost`

    Fill `accountNumber`, `accountRegion`, `bucketName` and `logGroup` with your account ID, region, any of your s3 buckets an a CloudWatch log group.

    Edit `  task_resource_defaults:`

    ```
           limits:
    		storage: 2000Mi
    ```

    to

    ```
           limits:
    		cpu: 16
             memory: 128Gi
             storage: 3Gi
    ```

    Edit `cluster_resource_manager` to look like this:

    ```
    cluster_resource_manager:
      # -- Enables the Cluster resource manager component
      enabled: true
      config:
        cluster_resources:
          customData:
            - x2:
                - projectQuotaCpu:
                    value: "2"
                - projectQuotaMemory:
                    value: "128Gi"
                - defaultIamRole:
                    value: "arn:aws:iam::{{ .Values.userSettings.accountNumber }}:role/flyte-user-role"
            - x4:
                - projectQuotaCpu:
                    value: "4"
                - projectQuotaMemory:
                    value: "128Gi"
                - defaultIamRole:
                    value: "arn:aws:iam::{{ .Values.userSettings.accountNumber }}:role/flyte-user-role"
            - x8:
                - projectQuotaCpu:
                    value: "8"
                - projectQuotaMemory:
                    value: "128Gi"
                - defaultIamRole:
                    value: "arn:aws:iam::{{ .Values.userSettings.accountNumber }}:role/flyte-user-role"
            - x16:
                - projectQuotaCpu:
                    value: "16"
                - projectQuotaMemory:
                    value: "128Gi"
                - defaultIamRole:
                     value: "arn:aws:iam::{{ .Values.userSettings.accountNumber }}:role/flyte-user-role"
                     
    ```

22. Edit `flyte\charts\flyte\values.yaml`

    from:

    ```
    # -- Docker image tag
         tag: v0.6.53 # FLYTEADMIN_TAG
    ```

    to:

    ```
    # -- Docker image tag
         tag: v0.6.33 # FLYTEADMIN_TAG
    ```

    *There was a bug in v0.6.53, do not change the version if you know that it's fixed

    

23. ```
    opta apply -c env.yaml
    ```

    This command can fail first time, if it does, try entering it again

24. ```
    opta apply -c flyte.yaml
    ```

25. ```
    aws eks --region us-east-2 update-kubeconfig --name opta-flyte
    ```

26. ```
    kubectl get service  -n flyte | grep flyteadmin
    ```

    Copy address

27. ```
    flytectl config init --host=<FLYTEADMIN_URL>:81 --storage --insecure
    ```

    Paste copied address instead of `<FLYTEADMIN_URL>`

28. ```
    export FLYTECTL_CONFIG=~/.flyte/config.yaml
    ```

29. Edit `~/.flyte/config.yaml`

    ```
    region: us-east-2
    ```

    ```
    container: <S3_BUCKET>
    ```

    `<S3_BUCKET>` is the bucket in  `values-eks.yaml`

30. ```
    kubectl get ingress -n flyte
    ```

    Copy address

31. Paste `<address>/console` in browser 

32. ```
    cd ../myflyteapp
    ```

33. ```
    docker build . --tag <DOCKERHUB_LOGIN>/flyte-test:001
    ```

34. ```
    docker push <DOCKERHUB_LOGIN>/flyte-test:001
    ```

35. ```
    pyflyte --pkgs myapp.workflows package -f --image <DOCKERHUB_LOGIN>/flyte-test:001
    ```

36. ```
    flytectl register files --project x4 --domain x4 --archive flyte-package.tgz --version v1 --logger.level=6
    ```

37. Launch the workflow in browser

38. To destroy a cluster:

    In `flyte/opta` directory

    ```
    opta destroy -c flyte.yaml --auto-approve
    opta destroy -c env.yaml --auto-approve
    ```

