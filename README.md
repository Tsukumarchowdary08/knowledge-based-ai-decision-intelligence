# Knowledge-Based Decision Systems from AI Data

Reproducible proof of concept for **Knowledge-Based Decision Systems from AI Data: Digital Transformation, Innovation, and Entrepreneurial Judgment**.

This repository uses [UCI Phishing Websites](https://doi.org/10.24432/C51W2X) (11,055 cases; 30 features) to test uncertainty-calibrated AI and cost-aware review allocation.

## Run on GitHub

Open the [Actions tab](https://github.com/Tsukumarchowdary08/knowledge-based-ai-decision-intelligence/actions/workflows/reproduce.yml), select **Reproduce research experiment**, and choose **Run workflow**. GitHub downloads the public UCI dataset, runs `experiment.py`, and publishes the two result tables, two PDF figures, and JSON summary as a downloadable **research-results** artifact.

To run locally: install `requirements.txt`, place the original `Training Dataset.arff` at `data/Training Dataset.arff`, then run `python experiment.py`.

## Research design

Fixed stratified train/validation/test split (60/20/20), three calibrated classifiers, model selection using validation Brier score, held-out test evaluation, and bootstrap interval for the review-policy cost difference.

**Limitations:** Review is assumed perfect, and costs are illustrative scenario units (miss=20, false alarm=2, review=0.6). The historical dataset does not demonstrate actual organizational innovation or real-world analyst outcomes. The original dataset is fetched from UCI rather than committed to the repository.

**Dataset citation:** Mohammad, R., & McCluskey, L. (2012). *Phishing Websites* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X
