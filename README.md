# LLM inference app

# Run
- depending on your installation case, you'll need to specify at least LLM name and resourse requests. For this, you could create a customized values file (see example at [/charts/llm-inference-app/values-mistral-onprem.yaml](/charts/llm-inference-app/values-mistral-onprem.yaml) )
- `helm upgrade llm charts/llm-inference-app --install --namespace llm-inference --create-namespace --values charts/llm-inference-app/values.yaml --values charts/llm-inference-app/values-vicuna-onprem.yaml` -- to install app, in this case, deploying Vicuna LLM in onprem cluster.

# Limitations
- StorageClass should support RWX mode in order to be able to mount into the multiple server pods.

## TODO:
1. Expose:
   1. domain name wildcard for inference *.apps.imdc3.org.neu.ro that points to cluster
   2. traefik ingress needs to watch other namespaces too. 
<!-- RAY setup:
1. First node -- RAY head, then ->
2. start all other worker nodes and connect to head, then ->
3. start LLM inference server on first node -- RAY head

scaling -- just changing number of workers 

https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html
this might be handy too
-->
