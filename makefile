# Docker image configuration
IMAGE_NAME ?= llm-inference
IMAGE_TAG ?= latest

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Build the Docker image in Minikube's environment
.PHONY: build-minikube
build-minikube:
	eval $$(minikube docker-env) && docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Push the image to Minikube's local cache
.PHONY: push-minikube
push-minikube:
	minikube cache add $(IMAGE_NAME):$(IMAGE_TAG)

# Remove the Docker image
.PHONY: clean
clean:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

# Build with no cache
.PHONY: rebuild
rebuild:
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .