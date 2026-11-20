MP282 -- Rooftop Solar Energy Prediction (MLOps)
Project Details
Field           Details
Group ID        G058
Problem ID      MP282
Project Title   Rooftop Solar Energy Prediction
Client          Dehradun Rooftop Solar Cooperative (simulated)
1. Problem Statement
The Dehradun Rooftop Solar Cooperative needs an automated system to
predict the daily energy generated (energy_kwh) by each rooftop solar
panel installation. These predictions can help cooperative members plan
high-energy activities, such as running pumps, and identify
under-performing solar installations.

The project aims to develop an MLOps pipeline for rooftop solar energy
prediction. The work includes data checks and preparation, followed by
model development and deployment activities in later stages.
2. Team Members and Responsibilities
The following team details and assigned responsibilities are taken from
the project PRD.
  Team member       Enrolment number  Assigned role     Responsibilities
  Vedikavyas        S24CSEU2341       Data Lead         Implement the
seven data-problem
detectors, DVC
tracking, cleaning
pipeline, and
update
findings.yaml.

  Aryan Sachan      S24CSEU2352       Model Lead        Build training
scripts, MLflow
logging, model
selection, v2
retraining, and
the promotion
rule.

  Aadarsh Dangi     S24CSEU2332       Service Lead      Implement FastAPI,
request
validation,
Dockerfile,
container
entrypoint, and
API tests.

  Apoorv Verma      S24CSEU2350       Ops Lead          Set up the CI
workflow,
Kubernetes
manifests, logging
format,
monitoring/PSI
scripts, and
deployment
automation.

Mansi Tripathi    S24CSEU0255       Testing & CI Lead Write pytest
suites, set up the
GitHub Actions CI
workflow, enhance
the Dockerfile,
maintain
documentation
scripts, and
prepare the
project report.

3. Stage 1 -- Repository Setup and Data Checks
Stage 1 status: Completed
During Stage 1, the focus was on identifying data-quality issues,
documenting the findings, writing Python checks, and tracking the
datasets using DVC.
Work completed
1. Checked all seven data issues
   Examined the dataset for each of the seven specified data-problem
   types.
2. Updated findings.yaml
   Recorded the findings from the data checks in the findings.yaml
   file.
3. Implemented Python data-checking code
   Wrote Python code to check and identify the seven data issues.
4. Tracked datasets using DVC
   Used Data Version Control (DVC) to track the datasets and maintain
   data versioning.
5. Pushed work to GitHub
   Committed and pushed the Stage 1 work to the project's GitHub
   repository.
Stage 1 deliverables
- Python code for checking the seven data issues.
- Updated findings.yaml file containing the data-check findings.
- Datasets tracked using DVC.
- Stage 1 work pushed to GitHub.

5. Stage 1 Summary
Stage 1 established the initial data-validation and version-control work
for the MP282 Rooftop Solar Energy Prediction project. All seven
specified data issues were checked, the findings were recorded in
findings.yaml, Python code was written to perform the checks, datasets
were tracked using DVC, and the completed work was pushed to GitHub.
Group ID: G058
Problem ID: MP282
Project: Rooftop Solar Energy Prediction (MLOps)
