## Metric

<details><summary>Configure</summary>
<p>

In this repository the Intersection over Union (**mean_IoU**) and the Dice score (**mean_Dice**) is provided.
Both metric update a confusion matrix in each step and compute a final scores at the end of each epoch.
The final score is composed of first calculating the score for each class and then taking the class wise mean.
By default only this final score is returned and logged, if additionally the score for each class is required use  **mean_IoU_Class** or **mean_Dice_Class**.
Some additional configurations are provided, adopt them in the config files or override them from
commandline:
- **name**: Name of the metric that should be optimized. The name has to be one the metrics which is logged to tensorboard.
If a step-wise or image-wise computed metric should be optimized, the "_stepwise" or "_per_image" postfix has to be used (e.g. meanDice_stepwise).
If the metric should be optimized for a single class, add the class name (for mean_Dice_Class and mean_IoU_Class) e.g. meanDice_class1. 
- **metric_global**: The metric is updated in each step and computed once at the end of each epoch. 
True by default.
- **metric_per_img**: The metric is computed for each image and averaged at the end of each epoch (avg. over all images).
False by default. Can be combined with *metric_global*.
- **train_metric**: True or False (False by default), provides the possibility to have a separate metric during
  training.

````shell
python training.py metric=mean_IoU         # mean Intersection over Onion (IoU)
python training.py metric=mean_Dice        # mean Dice score
python training.py metric=mean_IoU_Class   # mean IoU with additionally logging scores for each class
python training.py metric=mean_Dice_Class  # mean Dice with additionally logging scores for each class
````

</p>
 </details>

<details><summary>Customize</summary>
<p>

For defining a new metric use
the [torchmetric](https://torchmetrics.readthedocs.io/en/stable/pages/implement.html) package.
This makes the metric usable for multi-GPU training, a python dummy for such a metric can be found
below.
More information on how to define a torchmetrics can be
found [here](https://torchmetrics.readthedocs.io/en/stable/pages/implement.html)
As a restriction in this repository, the *compute()* method must return either a single tensor or a
dict.
A dict should be used when multiple metrics are returned, e.g. the IoU for each class separately.  
If a dict is used the metric is logged named by the corresponding key (avoid duplicates), if a single tensor is
returned it will be named by name of the metric defined in the config.

````py
from torchmetrics import Metric

class CustomMetric(Metric):
    def __init__(self, ...):
        ...
        #define state variables like this to make your metric multi gpu usable
        self.add_state("variable", default=XXX, dist_reduce_fx="sum", )
        ...

    def update(self, pred: torch.Tensor, gt: torch.Tensor):  
        # input is the batch-wise ground truth (gt) and models prediction (pred)
        ...     # pred.shape= [batch_size, num_classes, height, width]
                # gt.shape= [batch_size, height, width]

    def compute(self):
        ...  # do your computations
        return metric  # return the metric which should be optimized
        # or
        return {"metric1":value,"metric2":value,...} # if you want additional metrics to be logged 
                                                     # return them in dict format as a second arguments
    
    # (Optional) This function can be used to save metric states, e.g. a confusion matrix.
    # If the metric has a save_state function, the function is called in validation_epoch_end
    # If you dont need this functionality you don't need to define this function
    def save_state(self, trainer: pl.Trainer):
        # Save whatever you want to save
        ...
        
````

After implementing the metric you have to set up the config of the metric.
Therefore create a *my_metric.yaml* in *config/metric/* and use the following dummy to define the
metric.
*name* should be the name of your target metric which should be one of the metrics defined in
metrics(if the metric returns a single tensor), if the metric returns a dict *name* should be a key in this dict.
The remaining Parameters should be set as described in the *Configure* section above

`````yaml
config/metric/my_metric.yaml
─────────────────────────────
name: mymetric_name          # which metric to optimize - should be on of the names defined in METRIC.METRICS
train_metric: False    # If also a train metric is wanted (in addition to a validation metric)
metric_global: True      # If True metric is updated in each step and computed once at the end of the epoch
metric_per_img: False    # If True metric is computed for each image and averaged over all images - exclusively with call_stepwise but can be combined with call_global.

metrics:
    mymetric_name: # define the name of the metric, needed for logging and to find the target metric
      _target_: src.metric.myMetricClass  # path to the metric Class
      ...
      #num_classes: ${DATASET.NUM_CLASSES}  # list of arguments for initialization, e.g. number of classes
`````

````shell
python training.py metric=my_metric
````

</p>
</details>