# OpenShift Clients

The OpenShift client `oc` simplifies working with Kubernetes and OpenShift
clusters, offering a number of advantages over `kubectl` such as easy login,
kube config file management, and access to developer tools. The `kubectl`
binary is included alongside for when strict Kubernetes compliance is necessary.

To learn more about OpenShift, visit [docs.openshift.com](https://docs.openshift.com)
and select the version of OpenShift you are using.

## Installing the tools

After extracting this archive, move the `oc` and `kubectl` binaries
to a location on your PATH such as `/usr/local/bin`. Then run:

    oc login [API_URL]

to start a session against an OpenShift cluster. After login, run `oc` and
`oc help` to learn more about how to get started with OpenShift.

## Build and Run using podman
```
podman build -t finetune:v0.2 .

mkdir -p cache
chown -R 1001:1001 cache
podman run -e HF_TOKEN='<huggingface-token>' -v ${PWD}/cache/:/opt/app-root/src/.cache finetune:v0.2
```

## License

OpenShift is licensed under the Apache Public License 2.0. The source code for this
program is [located on github](https://github.com/openshift/oc).
# llamaModelsOnOpenShift
