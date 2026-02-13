# Topic 6: Autoscaling

## What You'll Learn

This topic teaches you how to:
- Set up Horizontal Pod Autoscaling (HPA)
- Scale based on CPU/memory metrics
- Scale based on custom metrics (request rate, latency)
- Configure scaling policies
- Handle scale-up and scale-down events

## Why Autoscaling?

### Benefits
- **Cost optimization**: Scale down when not needed
- **Performance**: Scale up under load
- **Automation**: No manual intervention needed
- **Efficiency**: Right-size resources

### When to Scale
- **High CPU/memory**: Pods are overloaded
- **High request rate**: Many incoming requests
- **High latency**: System struggling
- **Queue building**: Requests waiting

## Types of Autoscaling

### 1. Horizontal Pod Autoscaler (HPA)
Scales number of pod replicas based on metrics.

### 2. Vertical Pod Autoscaler (VPA)
Adjusts resource requests/limits (advanced).

### 3. Cluster Autoscaler
Adds/removes nodes (cloud providers).

## HPA Basics

### Simple HPA (CPU-based)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### How It Works
1. HPA checks metrics every 15s (default)
2. Compares current vs target
3. Calculates desired replicas
4. Updates deployment
5. K8s creates/destroys pods

## Custom Metrics

### Request Rate Scaling
Scale based on requests per second:
```yaml
metrics:
- type: Pods
  pods:
    metric:
      name: http_requests_per_second
    target:
      type: AverageValue
      averageValue: "10"
```

### Latency-based Scaling
Scale when latency is high:
```yaml
metrics:
- type: Pods
  pods:
    metric:
      name: request_latency_p95
    target:
      type: AverageValue
      averageValue: "500m"  # 500ms
```

## Scaling Behavior

### Scale-up Policy
```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 0
    policies:
    - type: Percent
      value: 100
      periodSeconds: 15
    - type: Pods
      value: 4
      periodSeconds: 15
    selectPolicy: Max
```

### Scale-down Policy
```yaml
behavior:
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
    - type: Percent
      value: 50
      periodSeconds: 60
    selectPolicy: Min
```

## Metrics Server

HPA needs metrics. Install metrics server:
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

Verify:
```bash
kubectl top nodes
kubectl top pods
```

## Prometheus Adapter

For custom metrics, use Prometheus Adapter:
1. Install Prometheus
2. Install Prometheus Adapter
3. Configure custom metrics
4. HPA can query Prometheus

See `08_monitoring/` for Prometheus setup.

## Complete HPA Example

See `hpa.yaml` for a complete example with:
- CPU scaling
- Memory scaling
- Custom metrics
- Scaling policies

## Testing Autoscaling

### Generate Load
```bash
# Use Locust or similar
locust -f locust_test.py --host=http://<service-url> \
  --users 50 --spawn-rate 5
```

### Watch Scaling
```bash
# Watch HPA
kubectl get hpa -w

# Watch pods
kubectl get pods -w

# Watch deployment
kubectl get deployment llm-serving -w
```

### Check Metrics
```bash
# CPU/Memory
kubectl top pods

# HPA events
kubectl describe hpa llm-serving-hpa
```

## Best Practices

1. **Set min replicas**: Always have some pods running
2. **Set max replicas**: Prevent runaway scaling
3. **Use stabilization windows**: Avoid flapping
4. **Monitor scaling events**: Track scale-up/down
5. **Test under load**: Verify scaling works
6. **Consider costs**: More pods = more cost

## Common Issues

### HPA Not Scaling
- Check metrics server is running
- Verify metrics are available
- Check HPA status: `kubectl describe hpa`
- Ensure deployment has resource requests

### Scaling Too Aggressively
- Increase stabilization window
- Adjust scaling policies
- Check for metric spikes

### Scaling Too Slowly
- Decrease stabilization window
- Adjust scaling policies
- Check metric collection delay

## Exercises

1. **Basic HPA**: Set up CPU-based autoscaling
2. **Custom metrics**: Scale based on request rate
3. **Scaling policies**: Configure scale-up/down behavior
4. **Load test**: Generate load and observe scaling
5. **Monitor**: Track scaling events and metrics

## Next Steps

- **Topic 7**: Canary deployments for safe updates
- **Topic 8**: Monitor autoscaling with Grafana
- **Topic 9**: Version management with scaling

## Further Reading

- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Custom Metrics](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
- [Scaling Best Practices](https://kubernetes.io/docs/concepts/workloads/autoscaling/)

