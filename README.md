# DeepEutaxy

DeepEutaxy: Diversity in Weight Search Direction for Fixing Deep Learning Model Training through Batch Prioritization

## Usage

Use the following command to run the code:

Format:
```
python <methods>.py -m <model> -d <dataset> -v <version> -c <clr>
```
* `methods`: `DE`, `SPL`, or `BL`
* `model`: `cnn`, `lenet5`, `resnet20`, `resnet32`, `mobilenetv2`
* `dataset`: `mnist` and `cifar10`
* `version`: version number
* `clr`: `0` does not Cyclical Learning Rate, or `1` uses Cyclical Learning Rate

