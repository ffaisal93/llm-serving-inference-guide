# Topic 9: Model Versioning

## What You'll Learn

This topic teaches you how to:
- Manage multiple model versions
- Implement model registry
- Version models semantically
- Track model metadata
- Enable quick rollbacks
- Support A/B testing

## Why Model Versioning?

### Benefits
- **Reproducibility**: Know exactly which model is running
- **Rollback**: Quickly revert to previous version
- **A/B testing**: Compare model versions
- **Audit trail**: Track model changes
- **Compliance**: Meet regulatory requirements

### Challenges
- **Storage**: Multiple versions take space
- **Complexity**: Managing versions adds overhead
- **Testing**: Need to test each version
- **Deployment**: Coordinate version updates

## Versioning Strategies

### 1. Semantic Versioning
```
v1.0.0  # Major.Minor.Patch
v1.1.0  # Minor update
v2.0.0  # Major update
```

### 2. Git-based
Use Git tags/commits for versions.

### 3. Timestamp-based
```
model-2024-01-15-10-30-00
```

### 4. Hash-based
Use model hash as version identifier.

## Model Registry

### What to Store
- **Model files**: Weights, tokenizer, config
- **Metadata**: Training date, metrics, dataset
- **Code**: Training script, preprocessing
- **Environment**: Dependencies, requirements

### Storage Options
- **S3/GCS**: Object storage
- **HuggingFace Hub**: Model hosting
- **MLflow**: Model registry
- **Local filesystem**: For development

## Implementation

### Simple File-based Registry
```
models/
  v1.0.0/
    model.bin
    tokenizer.json
    config.json
    metadata.json
  v1.1.0/
    model.bin
    tokenizer.json
    config.json
    metadata.json
```

### Metadata Schema
```json
{
  "version": "v1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "model_name": "gpt2",
  "training_date": "2024-01-10",
  "metrics": {
    "accuracy": 0.95,
    "latency_ms": 150
  },
  "dataset": "dataset-v1",
  "git_commit": "abc123"
}
```

## Version Management API

### List Versions
```python
GET /api/v1/models/versions
```

### Get Version Info
```python
GET /api/v1/models/versions/v1.0.0
```

### Deploy Version
```python
POST /api/v1/models/versions/v1.0.0/deploy
```

### Rollback
```python
POST /api/v1/models/rollback
{
  "target_version": "v1.0.0"
}
```

## Integration with Serving

### Environment Variable
```yaml
env:
- name: MODEL_VERSION
  value: "v1.0.0"
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  model_version: "v1.0.0"
  model_path: "/models/v1.0.0"
```

### Dynamic Loading
Load model version at runtime based on config.

## Rollback Procedure

### Quick Rollback
1. Identify current version
2. Identify target version
3. Update deployment
4. Verify health
5. Monitor metrics

### Automated Rollback
Set up alerts that trigger rollback:
- Error rate spike
- Latency increase
- Quality degradation

## A/B Testing Support

### Deploy Multiple Versions
```yaml
# Version A
deployment-a:
  model_version: "v1.0.0"
  
# Version B  
deployment-b:
  model_version: "v1.1.0"
```

### Compare Metrics
Track metrics per version:
- Performance (latency, throughput)
- Quality (accuracy, user feedback)
- Cost (GPU hours)

## Best Practices

1. **Version everything**: Models, code, configs
2. **Tag releases**: Use semantic versioning
3. **Store metadata**: Track training info
4. **Test before deploy**: Validate new versions
5. **Document changes**: Changelog for each version
6. **Automate**: Use CI/CD for versioning

## Exercises

1. **Create registry**: Set up model version storage
2. **Version model**: Tag and store model version
3. **Deploy version**: Deploy specific version
4. **Rollback**: Practice rolling back
5. **A/B test**: Compare two versions

## Next Steps

- **Topic 7**: Use versions in canary deployments
- **Topic 10**: Detect drift per version
- **Topic 8**: Monitor version performance

## Further Reading

- [Semantic Versioning](https://semver.org/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Model Versioning Best Practices](https://www.mlflow.org/docs/latest/models.html)

