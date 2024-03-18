# LLM inference app

# Limitations
- StorageClass should support RWX mode in order to be able to mount into the pod.

## TODO:
1. domain name wildcard for inference? *.apps.imdc3.org.neu.ro that points to cluter, that routes requests via ingress to needed app services
2. traefik ingress (re-)configuration -- we need to watch other namespaces too. Probably, just enable all?
3. STS instead of Deployment? (if we cannot create RWX PVs, this will retain PVC&PV for caching)
