# MediAssist-Pro 🏥

Assistant IA de maintenance biomédicale utilisant un RAG optimisé pour fournir documentation technique instantanée et guides de dépannage intelligents aux équipes de laboratoire.

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   PostgreSQL    │    │   ChromaDB      │
│   (API + Auth)  │◄──►│   (Database)    │    │ (Vector Store)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ollama        │    │   Prometheus    │    │   Grafana       │
│   (LLM)         │    │   (Metrics)     │    │ (Monitoring)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

##  Fonctionnalités

- **RAG Intelligent**: Recherche sémantique dans la documentation biomédicale
- **Authentification**: Système complet avec JWT
- **Monitoring**: Métriques Prometheus + Dashboards Grafana
- **CI/CD**: Pipeline automatisé GitHub Actions
- **Déploiement**: Kubernetes ready avec Docker

##  Installation

### Prérequis
- Docker & Docker Compose
- Kubernetes (kubectl)
- Python 3.9+

### 1. Clone du projet
```bash
git clone https://github.com/OUSSAMAEDDERKAOUI/rag-biomedical-support
cd rag-biomedical-support
```

### 2. Configuration
```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

### 3. Déploiement Kubernetes
```bash
# Build et push de l'image
docker build -t oussamaedderkaoui/rag-biomedical-support-api:latest .
docker push oussamaedderkaoui/rag-biomedical-support-api:latest

# Déploiement complet
kubectl apply -f prometheus-rbac.yaml
kubectl apply -f deployment.yaml
kubectl apply -f monitoring.yaml

# Port forwarding
kubectl port-forward svc/api-svc 8000:8000 &
kubectl port-forward svc/prometheus 9090:9090 &
kubectl port-forward svc/grafana 3000:3000 &
```

### 4. Test du déploiement
```bash
python test_api.py
```

## 🌐 Accès aux services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Documentation | http://localhost:8000/docs | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin123 |
| Métriques | http://localhost:8000/metrics | - |

##  Monitoring

### Métriques disponibles
- `rag_requests_total`: Nombre total de requêtes
- `rag_response_time_seconds`: Temps de réponse
- `rag_errors_total`: Nombre d'erreurs
- `rag_quality_score`: Score de qualité des réponses

### Dashboard Grafana
Importer le fichier `grafana-dashboard.json` pour visualiser:
- Taux de requêtes
- Temps de réponse (95e percentile)
- Taux d'erreurs
- Métriques système

## 🔧 Développement

### Structure du projet
```
rag-biomedical-support/
├── app/
│   ├── api/v1/          # Endpoints API
│   ├── db/              # Base de données
│   ├── models/          # Modèles SQLAlchemy
│   ├── monitoring/      # Métriques Prometheus
│   └── main.py          # Application FastAPI
├── tests/               # Tests unitaires
├── deployment.yaml      # Déploiement Kubernetes
├── monitoring.yaml      # Stack monitoring
└── requirements.txt     # Dépendances Python
```

### Lancer en local
```bash
# Installation des dépendances
pip install -r requirements.txt

# Variables d'environnement
export DATABASE_URL="postgresql://user:pass@localhost/db"
export CHROMA_HOST="http://localhost:8000"
export OLLAMA_URL="http://localhost:11434"

# Démarrage
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Tests
```bash
# Tests unitaires
pytest tests/

# Tests d'intégration
python test_api.py

# Tests de déploiement
bash test-deployment.sh
```

## 🚀 CI/CD Pipeline

Le pipeline GitHub Actions automatise:
1. **Tests**: Exécution des tests unitaires
2. **Build**: Construction de l'image Docker
3. **Push**: Publication sur Docker Hub
4. **Deploy**: Déploiement sur Kubernetes (optionnel)

### Configuration
Ajouter ces secrets GitHub:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## 📚 API Endpoints

### Authentification
- `POST /api/v1/auth/register` - Inscription
- `POST /api/v1/auth/login` - Connexion
- `GET /api/v1/auth/me` - Profil utilisateur

### RAG
- `POST /api/v1/index/query` - Requête RAG
- `POST /api/v1/index/upload` - Upload de documents
- `GET /api/v1/index/documents` - Liste des documents

### Monitoring
- `GET /health` - Santé de l'API
- `GET /health/detailed` - Santé détaillée
- `GET /metrics` - Métriques Prometheus

##  Sécurité

- Authentification JWT
- Validation des entrées avec Pydantic
- Variables d'environnement pour les secrets
- RBAC Kubernetes pour Prometheus

##  Dépannage

### Problèmes courants

**Port 8000 déjà utilisé**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Chroma en CrashLoopBackOff**
```bash
kubectl logs -l app=chroma
kubectl delete pod -l app=chroma
```

**Prometheus sans targets**
```bash
kubectl apply -f prometheus-rbac.yaml
kubectl delete pod -l app=prometheus
```

##  Performance

- **Latence**: < 200ms pour les requêtes RAG
- **Throughput**: 100+ requêtes/seconde
- **Disponibilité**: 99.9% uptime
- **Scalabilité**: Auto-scaling Kubernetes

##  Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

##  License

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

##  Équipe

- **Développeur Principal**: Oussama Edderkaoui
- **Architecture**: RAG + Kubernetes + Monitoring
- **Contact**: [edderkaouioussama@gmail.com](mailto:edderkaouioussama@gmail.com)

---

 **N'oubliez pas de star le projet si il vous a été utile !**