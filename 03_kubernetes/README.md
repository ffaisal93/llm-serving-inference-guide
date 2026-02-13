# Topic 3: Kubernetes Deployment

## What You'll Learn

This topic teaches you how to:
- Deploy LLM serving applications to Kubernetes
- Configure health checks and probes
- Set resource limits and requests
- Use ConfigMaps and Secrets
- Handle rolling updates
- Scale applications

## Why We Need This

### Business Need
- **Cost optimization**: Auto-scale down during low traffic, save money
- **Reliability**: 99.9% uptime SLA requires automatic recovery
- **Speed**: Deploy updates in seconds, not hours
- **Compliance**: Resource limits ensure fair resource usage

### Technical Need
- **Orchestration**: Manage hundreds of containers automatically
- **Resource isolation**: Prevent one service from starving others
- **Service discovery**: Automatic load balancing across pods
- **Self-healing**: Restart failed containers automatically

### Real-World Impact
Without Kubernetes:
- ❌ Manual scaling (slow, error-prone)
- ❌ Downtime during deployments
- ❌ Resource conflicts between services
- ❌ Difficult to manage at scale (100+ services)

## Industry Use Cases

### 1. **Large-Scale ML Platforms**
**Company**: OpenAI, Anthropic, HuggingFace
**Use Case**:
- Serve millions of API requests per day
- Auto-scale based on demand
- Zero-downtime deployments

**Example:**
```yaml
# Handles traffic spikes automatically
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 10
  maxReplicas: 1000  # Scale to 1000 pods during peak
```

### 2. **Enterprise ML Infrastructure**
**Company**: Banks, healthcare, finance
**Use Case**:
- Multi-tenant ML serving
- Resource quotas per team
- Compliance and audit trails

**Example:**
```yaml
# Resource quotas per namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.nvidia.com/gpu: "10"
```

### 3. **Multi-Region Deployment**
**Company**: Global SaaS companies
**Use Case**:
- Deploy same model to multiple regions
- Low latency for global users
- Regional failover

**Example:**
```yaml
# Deploy to multiple regions
kubectl apply -f deployment.yaml --context us-east
kubectl apply -f deployment.yaml --context eu-west
kubectl apply -f deployment.yaml --context asia-pacific
```

### 4. **A/B Testing Infrastructure**
**Company**: Tech companies (Netflix, Spotify)
**Use Case**:
- Run multiple model versions simultaneously
- Split traffic between versions
- Compare performance

**Example:**
```yaml
# Deploy model A and B
kubectl apply -f model-a-deployment.yaml
kubectl apply -f model-b-deployment.yaml
# Route 50% traffic to each
```

### 5. **Development/Staging/Production**
**Company**: All companies
**Use Case**:
- Same deployment config for all environments
- Easy promotion from dev → staging → prod

**Example:**
```bash
# Deploy to dev
kubectl apply -f deployment.yaml -n dev

# Test, then promote to staging
kubectl apply -f deployment.yaml -n staging

# Finally, production
kubectl apply -f deployment.yaml -n production
```

## Industry-Standard Boilerplate Code

### Complete Production Deployment (Industry Standard)

```yaml
# deployment.yaml
# Used by: Production ML serving at scale
apiVersion: v1
kind: Namespace
metadata:
  name: llm-serving
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
  namespace: llm-serving
data:
  model_name: "gpt2"
  max_length: "100"
  temperature: "0.7"
  log_level: "INFO"
---
apiVersion: v1
kind: Secret
metadata:
  name: llm-secrets
  namespace: llm-serving
type: Opaque
stringData:
  api_key: "your-api-key-here"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
  namespace: llm-serving
  labels:
    app: llm-serving
    version: v1.0.0
spec:
  replicas: 3  # High availability
  selector:
    matchLabels:
      app: llm-serving
  template:
    metadata:
      labels:
        app: llm-serving
        version: v1.0.0
    spec:
      containers:
      - name: llm-serving
        image: your-registry/llm-serving:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        # From ConfigMap
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: llm-config
              key: model_name
        - name: MAX_LENGTH
          valueFrom:
            configMapKeyRef:
              name: llm-config
              key: max_length
        # From Secret
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: api_key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1  # GPU request
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1  # GPU limit
        # Health checks (critical for production)
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60  # Model loading time
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          failureThreshold: 30  # Allow 5 minutes for startup
          periodSeconds: 10
      # Node selector for GPU nodes
      nodeSelector:
        accelerator: nvidia-tesla-v100
      # Tolerations for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: llm-serving
  namespace: llm-serving
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: llm-serving
---
# Ingress for external access
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
  namespace: llm-serving
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - llm-api.yourcompany.com
    secretName: llm-tls
  rules:
  - host: llm-api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-serving
            port:
              number: 80
```

### Deployment Script (Industry Standard)

```bash
#!/bin/bash
# deploy.sh
# Used by: CI/CD pipelines, deployment automation

set -e

NAMESPACE="llm-serving"
IMAGE_TAG="${1:-latest}"
ENVIRONMENT="${2:-staging}"

echo "Deploying to ${ENVIRONMENT} with image tag ${IMAGE_TAG}"

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Update image in deployment
kubectl set image deployment/llm-serving \
  llm-serving=your-registry/llm-serving:${IMAGE_TAG} \
  -n ${NAMESPACE}

# Wait for rollout
kubectl rollout status deployment/llm-serving -n ${NAMESPACE} --timeout=5m

# Verify deployment
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}

echo "Deployment complete!"
```

### Rollback Script (Industry Standard)

```bash
#!/bin/bash
# rollback.sh
# Used when issues are detected in production

NAMESPACE="llm-serving"

echo "Rolling back deployment..."

kubectl rollout undo deployment/llm-serving -n ${NAMESPACE}
kubectl rollout status deployment/llm-serving -n ${NAMESPACE}

echo "Rollback complete!"
```

### Key Concepts
- **Pods**: Smallest deployable unit (your container)
- **Deployments**: Manage pod replicas
- **Services**: Expose pods to network
- **ConfigMaps**: Configuration data
- **Secrets**: Sensitive data
- **Ingress**: External access

## Prerequisites

### Local Kubernetes
```bash
# Option 1: Minikube
minikube start

# Option 2: Kind
kind create cluster

# Option 3: Docker Desktop (has K8s built-in)
```

### Verify Setup
```bash
kubectl cluster-info
kubectl get nodes
```

## Basic Deployment

### Deployment Manifest
See `deployment.yaml` for a complete example.

Key components:
- **Deployment**: Manages pods
- **Service**: Exposes pods
- **ConfigMap**: Configuration
- **Resource limits**: CPU/memory

### Deploy
```bash
kubectl apply -f deployment.yaml
```

### Check Status
```bash
kubectl get pods
kubectl get services
kubectl logs -f deployment/llm-serving
```

## Health Checks

### Liveness Probe
Detects if pod is alive. If fails, pod is restarted.
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10
```

### Readiness Probe
Detects if pod is ready to serve traffic.
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5
```

### Startup Probe
Gives pod time to start (useful for slow-starting apps).
```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  failureThreshold: 30
  periodSeconds: 10
```

## Resource Management

### Requests
Minimum resources guaranteed to pod.
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
```

### Limits
Maximum resources pod can use.
```yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1  # For GPU
```

### Why This Matters
- **Requests**: Scheduler uses this to place pods
- **Limits**: Prevents pods from consuming all resources
- **GPU**: Must specify for GPU workloads

## ConfigMaps and Secrets

### ConfigMap
Store non-sensitive configuration.
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
data:
  model_name: "gpt2"
  max_length: "100"
  temperature: "0.7"
```

### Secret
Store sensitive data (API keys, tokens).
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-secrets
type: Opaque
data:
  api_key: <base64-encoded>
```

### Using in Deployment
```yaml
env:
  - name: MODEL_NAME
    valueFrom:
      configMapKeyRef:
        name: llm-config
        key: model_name
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: llm-secrets
        key: api_key
```

## Scaling

### Manual Scaling
```bash
kubectl scale deployment llm-serving --replicas=3
```

### Auto-scaling (HPA)
See `06_autoscaling/` for detailed HPA setup.

## Rolling Updates

### Update Image
```bash
kubectl set image deployment/llm-serving \
  llm-serving=llm-serving:v2
```

### Rollback
```bash
kubectl rollout undo deployment/llm-serving
```

### Check Rollout Status
```bash
kubectl rollout status deployment/llm-serving
```

## GPU Support

### Node Labels
```bash
# Label nodes with GPU
kubectl label nodes <node-name> accelerator=nvidia-tesla-v100
```

### GPU Resource Request
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

### Device Plugin
Install NVIDIA device plugin:
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

## Ingress

### Expose Service Externally
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
spec:
  rules:
  - host: llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-serving
            port:
              number: 8000
```

## Monitoring

### View Logs
```bash
kubectl logs -f deployment/llm-serving
kubectl logs -f <pod-name>
```

### Describe Resources
```bash
kubectl describe pod <pod-name>
kubectl describe deployment llm-serving
```

### Get Events
```bash
kubectl get events --sort-by='.lastTimestamp'
```

## Common Issues

### Pod Not Starting
- Check logs: `kubectl logs <pod-name>`
- Describe pod: `kubectl describe pod <pod-name>`
- Check events: `kubectl get events`

### Out of Resources
- Check node resources: `kubectl describe node`
- Adjust resource requests/limits
- Add more nodes

### Image Pull Errors
- Check image name and tag
- Verify image registry access
- Check image pull secrets

### Health Check Failures
- Verify health endpoint works
- Adjust probe timing
- Check application startup time

## Best Practices

1. **Always set resource limits**: Prevent resource exhaustion
2. **Use health checks**: Enable automatic recovery
3. **Use ConfigMaps**: Don't hardcode configuration
4. **Tag images**: Use semantic versioning
5. **Test locally**: Use minikube/kind before production
6. **Monitor**: Set up logging and metrics

## Exercises

1. **Basic deployment**: Deploy app to local K8s
2. **Health checks**: Add liveness/readiness probes
3. **Resource limits**: Set appropriate CPU/memory
4. **ConfigMap**: Move config to ConfigMap
5. **Scaling**: Scale to 3 replicas
6. **Rolling update**: Update to new version

## Next Steps

- **Topic 4**: Load test your K8s deployment
- **Topic 6**: Set up auto-scaling
- **Topic 8**: Add monitoring

## Further Reading

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kubernetes for ML](https://kubernetes.io/blog/tags/machine-learning/)
- [GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

