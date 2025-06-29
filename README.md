
# Predictive Maintenance using Machine Learning and MLOps

## 🚀 Project Overview

This project is a production-grade **Predictive Maintenance System** that uses machine learning to detect potential machine failures based on sensor data and operational conditions. The complete pipeline follows **MLOps best practices** with experiment tracking, version control, model registry, deployment, and CI/CD integration.

---

## 📈 Problem Statement

Machines in manufacturing processes undergo wear and tear over time. Identifying the risk of failure before it occurs can:
- Reduce unplanned downtime
- Lower maintenance costs
- Improve overall efficiency and productivity

**Goal**: Predict machine failure (`machine failure` column) using historical sensor data.

---

## 💡 Business Impact

- 📉 **Cost Reduction**: Minimizes unplanned machine repairs and production halts.
- 🧠 **Smarter Decision Making**: Predict failures before they occur.
- ⏱ **Efficiency Boost**: Enables planned maintenance schedules and resource allocation.
- 📦 **Scalability**: Cloud-native, modular, and containerized architecture allows easy scaling and integration into existing systems.

---

## 🛠️ Tech Stack

| Category            | Tools Used                                                                 |
|---------------------|----------------------------------------------------------------------------|
| **Language**        | Python 3.10                                                                 |
| **IDE/Codebase**    | Cookiecutter Data Science Structure                                         |
| **Data Versioning** | DVC + DagsHub                                                               |
| **Experimentation** | MLflow + DagsHub                                                            |
| **Modeling**        | scikit-learn (Random Forest, etc.)                                         |
| **Tracking**        | MLflow (local + DagsHub remote backend)                                    |
| **Storage**         | S3 (AWS) for model registry + DVC remote                                   |
| **Deployment**      | Flask App (Dockerized) on EC2 with ECR + Self-hosted GitHub runner          |
| **CI/CD**           | GitHub Actions (CI/CD pipeline) + Docker + AWS                             |
| **Monitoring**      | Prometheus metrics exposed in Flask API                                    |

---

## 🔁 Workflow & Components

### 📁 1. Project Setup
- Cookiecutter template used to scaffold project.
- Modules installed via `requirements.txt`.

### 📊 2. Exploratory Data Analysis (EDA)
- Basic statistics, distribution plots, correlation analysis.
- Derived new features like temperature difference, torque per rpm.

### 🏗️ 3. Modular Pipeline Components

Each component is modularized with logging, exception handling, and unit tests.

1. **Data Ingestion**
   - Reads CSV, splits into train/test, stores in feature store.

2. **Data Validation**
   - Schema validation using `schema.yaml`.
   - Checks for missing values, correct data types, ranges.

3. **Data Transformation**
   - Custom transformers, outlier handling, encoding.
   - SMOTEENN applied to handle class imbalance.

4. **Model Trainer**
   - Trains classifier using GridSearchCV + custom scoring.
   - Stores model as `.pkl` via `dill`.

5. **Model Evaluation**
   - Compares new vs old model.
   - If performance improves beyond a threshold (e.g., 0.02 F1 gain), accepts model.

6. **Model Pusher**
   - Pushes model artifacts to AWS S3.

7. **Model Registration**
   - Logs model to MLflow with `staging` or `production` stage.

---

## 🧪 Experiment Tracking (MLflow + DagsHub)
- MLflow integrated with DagsHub remote tracking.
- All metrics (`f1`, `precision`, `recall`, `accuracy`, `AUC`) logged per run.
- Visual dashboard to compare model performance.

---

## 🧬 DVC Pipelines

- `dvc.yaml` created to run the entire ML pipeline as stages.
- Parameters like timestamps are managed in `params.yaml`.
- `run_pipeline.py` is used to update timestamp and execute `dvc repro`.

```bash
python run_pipeline.py  # Updates params.yaml and runs DVC
dvc push                 # Pushes data to remote S3 bucket
```

---

## ☁️ AWS Integration

| Service | Usage |
|--------|-------|
| S3     | Remote storage for DVC + ML models |
| ECR    | Docker container registry |
| EC2    | Deployed Flask App using Docker + Self-hosted runner |
| IAM    | Access management + GitHub Secrets for AWS CLI |

---

## 🧪 Unit Testing

Test cases added in `tests/` and `scripts/`. CI pipeline runs them on every commit.

---

## 🧱 Deployment

1. **Dockerized Flask App** inside `flask_app/`
2. Model loaded from MLflow model registry (`Staging/Production`).
3. App provides binary classification + prediction probabilities.
4. `prometheus_client` exposes model metrics.

---

## 🐳 Docker + AWS EC2 Deployment

- Dockerfile + .dockerignore prepared.
- Container deployed to EC2 (Ubuntu) via GitHub self-hosted runner.
- Flask app listens on port 5000.

---

## 🤖 CI/CD

CI/CD is managed through **GitHub Actions**:

| Stage         | Description |
|---------------|-------------|
| CI            | Install, test, lint, build Docker image |
| Push to ECR   | Push built image to AWS Elastic Container Registry |
| Deploy to EC2 | EC2 runner pulls and runs container |

---

## 🔒 Secrets Management

Secrets like:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `ECR_REPOSITORY`
- `DAGSHUB_TOKEN`

are stored as GitHub Secrets.

---

## 📈 Prometheus Metrics

Flask API exposes real-time model inference metrics via:
```
/metrics
```

---

## 🌍 How to Run

### Clone and Setup Environment
```bash
git clone <repo>
cd <repo>
conda create -n atlas python=3.10
conda activate atlas
pip install -r requirements.txt
```

### Run ML Pipeline
```bash
python run_pipeline.py
```

### Start Local App
```bash
cd flask_app
python app.py
# OR (better for production)
gunicorn --bind 0.0.0.0:5000 app:app
```

---

## ✅ Output

- Trained models saved in S3
- Metrics tracked in MLflow
- App deployed via Docker on EC2
- CI/CD automated using GitHub Actions + self-hosted runner

---

## 👨‍💻 Author

**MD Arsalan Arshad**  
Aspiring ML Engineer | MLOps | Cloud-Native AI Systems | Data Science

---

## 📄 License

This project is under the [MIT License](LICENSE).
