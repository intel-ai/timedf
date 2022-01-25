## Local Sandbox

The Flyte Sandbox is a fully standalone minimal environment for running Flyte. It runs as a single Docker container.

###    Windows (with GitBash as a default terminal)

## **Installation**

1. Install Flytekit

   > pip install flytekit --upgrade

2. Install Docker

   Refer to https://docs.docker.com/get-docker/

3. Install Flytectl

   > curl -s https://raw.githubusercontent.com/flyteorg/flytectl/master/install.sh | sudo bash -s -- -b /usr/local/bin

   Test installation:

   > flytectl version

## **Usage**

1. Clone repo with flytekit-python-template

   > cookiecutter https://github.com/flyteorg/flytekit-python-template.git --directory="simple-example" -c 005f8830448095a50e42c2e60e764d00fbed4eb8 && cd flyte_example

2. Start Docker daemon

3. Start a new sandbox 

   > flytectl sandbox start --source .

4. Setup flytectl sandbox config

   > flytectl config init

5. Build the image inside the Flyte-sandbox container

   > flytectl sandbox exec -- docker build . --tag "myapp:v1"

6. Package the workflow

   > pyflyte --pkgs myapp.workflows package --image myapp:v1 -f

7. Upload this package to the Flyte backend

   > flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v1

8. Visualize the registered workflow

   > flytectl get workflows --project flytesnacks --domain development myapp.workflows.example.my_wf --version v1 -o doturl

9. Launch an execution 

   using FlyteConsole:

   ​	Go to http://localhost:30081/console

   ​	Then, go to flytesnacks development, select a workflow and launch it.

   using command line:

   > flytectl get launchplan --project flytesnacks --domain development myapp.workflows.example.my_wf --latest --execFile exec_spec.yaml
   >
   > flytectl create execution --project flytesnacks --domain development --execFile exec_spec.yaml

   Sample output: `execution identifier project:"flytesnacks" domain:"development" name:"ffa5d27577e1e4a3e886"` name is the \<execname>

   > flytectl get execution --project flytesnacks --domain development \<execname>

## Teardown

1. Teardown flyte sandbox

   > flytectl sandbox teardown



