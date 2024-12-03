#! /bin/bash
minikube start --driver docker --container-runtime docker --gpus all --memory 10g --cpus 6 && \
make build-minikube && \
cd charts && \
helm install vllm llm-inference-app --set env.HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
