# ISDN3000C_Lab03: RDK X5 AI Vision Challenge Template

This is the official template repository for the Lab03 of ISDN3000C.

This repository contains the necessary files to run a simple AI image classification task on your RDK X5.

## Repository Contents

*   `classify.py`: A Python script that uses the **PyTorch** library to load a pre-trained AI model (`ResNet18`), analyze the `sample_image.png`, and print the classification result.
*   `requirements.txt`: libraries to install before running `classify.py`.
*   `classify_batch.py`: for advanced task.
*   `sample_image.png`: A sample image of a  used as input for the AI model.

## Instructions

1.  **Fork this Repository**: Create your own copy of this repository on GitHub.
2.  **Clone it to your RDK**: Follow the assignment instructions to install `git` and clone **your forked repository** onto the RDK X5.
3.  **Install Dependencies**: As per the assignment, ensure you have installed the required Python libraries on your RDK:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Script**: Navigate into the repository directory on your RDK and execute the script:
    ```bash
    python classify.py
    ```
5.  **Use the Output**: The script will print the AI's prediction. Use this output to complete the final steps of the assignment (creating the webpage and submitting your work).
# ISDN3000C Lab03: RDK X5 AI Vision Challenge

## Basic Task
- Used ResNet18 to classify `sample_image.png`.
- Ran `python classify.py` and obtained a prediction, for example:
- Displayed the result in the webpage `task1/index.html`.
- Screenshot is saved as `result.png`.

## Advanced Task: Batch Image Classification
- Used `classify_batch.py` to classify all images in the `test_images/` folder.
- Results were written to `results.csv`.
- Example output:
- Full results are available in `results.csv`.

## Repository Contents
- `classify.py`: Script for single image classification.
- `classify_batch.py`: Script for batch classification and CSV output.
- `requirements.txt`: Python dependencies.
- `sample_image.png`: Example input image.
- `task1/index.html`: Webpage for displaying classification result.
- `result.png`: Screenshot of the webpage (basic task).
- `results.csv`: Batch classification results (advanced task).
- `test_images/`: Test images (ignored in `.gitignore`).

## Group Members
- Zhang Haochun (Student ID: 21147459)
- Shen Yuming （Student ID：20945165）[Other group members if applicable]
