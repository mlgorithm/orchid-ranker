# Terraform Module (Reference)

```hcl
module "orchid_ranker" {
  source = "git::https://github.com/farhad.vadiee/orchid-ranker//deploy/terraform/module"

  name            = "orchid-ranker"
  namespace       = "learning"
  image           = "ghcr.io/your-org/orchid-ranker:latest"
  replicas        = 2
  metrics_enabled = true

  env = {
    ORCHID_LOG_LEVEL      = "INFO"
    ORCHID_AUDIT_ENDPOINT = var.audit_endpoint
  }
}
```

This repository ships only documentation placeholders. Use the Helm chart under `deploy/helm` in conjunction with Terraform Helm releases for production deployments.
