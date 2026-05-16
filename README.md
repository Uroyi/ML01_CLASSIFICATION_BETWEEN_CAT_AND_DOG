# Cat vs Dog Classification with PyTorch

This project implements a binary image classification task to distinguish between cats and dogs using the PyTorch framework.

## Project Structure

- `dataset/`: Contains the image data.
    - `Cat/`: Images of cats.
    - `Dog/`: Images of dogs.
- `train.py`: The main script for training the model.
- `requirements.txt`: List of dependencies required for the project.
- `cat_dog_model.pth`: (Generated after training) The saved model weights.

## Dataset

The dataset used in this project is sourced from Kaggle:
[Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

It consists of thousands of images of cats and dogs organized into subfolders.

## Installation

1.  Clone this repository or navigate to the project directory.
2.  Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the training process, run the following command:

```bash
python train.py
```

### What `train.py` does:
- Loads the dataset from the `dataset/` directory.
- Applies data augmentation and normalization.
- Splits the data into training (80%) and validation (20%) sets.
- Uses a pretrained **ResNet18** model as the base.
- Trains for 10 epochs (configurable) using the Adam optimizer and Cross-Entropy loss.
- Saves the best model state to `cat_dog_model.pth`.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Pillow
- tqdm

These are specified in the `requirements.txt` file.

## License

This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.