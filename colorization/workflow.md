Docker: (r.c.videogorillas.com/colorize:docker)
```bash
docker run --rm -ti -v /input_folder/:/in/ -v /output_folder/:/out/ r.c.videogorillas.com/colorize:docker 
```

Kubernetes: (r.c.videogorillas.com/colorize:kube-1)
```bash
kubectl create namespace colorization
kubectl -n colorization create -f kube.yaml
```