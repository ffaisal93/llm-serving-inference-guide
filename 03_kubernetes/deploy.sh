#!/bin/bash
# Deployment script for Kubernetes

set -e

NAMESPACE="llm-serving"

echo "Deploying LLM serving to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Apply deployment
kubectl apply -f deployment.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/llm-serving -n ${NAMESPACE}

# Get service info
echo ""
echo "Deployment complete!"
echo ""
echo "Service endpoints:"
kubectl get svc -n ${NAMESPACE}

echo ""
echo "Pods:"
kubectl get pods -n ${NAMESPACE}

echo ""
echo "To access the service:"
echo "  kubectl port-forward -n ${NAMESPACE} svc/llm-serving 8000:80"
echo ""
echo "Or use:"
echo "  kubectl get svc -n ${NAMESPACE}"

