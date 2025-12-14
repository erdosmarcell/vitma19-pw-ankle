# Deep Learning Class (VITMMA19) Project Work
## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Erdős Marcell
- **Aiming for +1 Mark**: No

### Solution Description

In this project, I developed two different deep learning image models to classify ankle positions (Neutralis, Pronacio, Supinacio) from posterior view images, and then compared their performance.
The first model, a baseline named "TinyNet" (referencing its size), implements a simple, custom-designed convolutional neural network (CNN). The second model is based on ResNet18, utilizing a pre-trained network. For the ResNet model, the main network layers were frozen, and only the final fully connected output layer was mapped and trained for the specific three-class task.
The models were trained using the Adam optimizer and the cross-entropy loss function. The results show that the ResNet model fundamentally performs better than the other regarding accuracy, though labeling uncertainties prevent the achievement of high validation scores. It is worth noting that due to resource constraints, training and testing were performed on a CPU. Data preprocessing was completed, and a grouping method was used on the consensus images to identify a suitable number of accurately labeled images due to mixed-error annotations.

### Extra Credit Justification

[If you selected "Yes" for Aiming for +1 Mark, describe here which specific part of your work (e.g., innovative model architecture, extensive experimentation, exceptional performance) you believe deserves an extra mark.]

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t vitma19-pw-ankle .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v D:\BME\Melytanulas\vitma19-pw-ankle\data:/app/data vitma19-pw-ankle > log/run_log.txt 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01_data_processing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02_training.py`: The main script for defining the model and executing the training loop.
    - `03_evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04_inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation. (executed notebooks in data/results)
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
