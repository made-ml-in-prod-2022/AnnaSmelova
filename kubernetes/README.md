Kubernetes

Kubernetes установлен локально через  **minikube**

### Используемые образы:

    annasmelova/online_inference:v1
    annasmelova/online_inference:v2

Второй образ с задержкой 25 секунд при старте и с прерыванием работы через 120 секунд.

### Проверка, что кластер поднялся:

    kubectl cluster-info
  
### Деплой приложений в кластер:

    kubectl apply -f kubernetes/online-inference-pod.yaml
    kubectl apply -f kubernetes/online-inference-pod-resources.yaml
    kubectl apply -f kubernetes/online-inference-pod-probes.yaml
    kubectl apply -f kubernetes/online-inference-replicaset.yaml
    kubectl apply -f kubernetes/online-inference-deployment-blue-green.yaml
    kubectl apply -f kubernetes/online-inference-deployment-rolling-update.yaml
  
### Проверка, что все поднялось:

    kubectl get pods
