## LR Scheduler

<details><summary>Configure</summary>
<p>

Currentyl the following schedulers are supported and can be used as shown below.
By default the polynomial scheduler is used (stepwise):

- polynomial: Polynomial lr scheduler over the number of steps: *(1-current_step/max_step)^0.9*
- polynomial_epoch: Polynomial lr scheduler over number of epochs: *(1-current_epoch/max_epoch)^0.9*

````shell
python training.py lr_scheduler=polynomial
python training.py lr_scheduler=polynomial_epoch
````

</p>
</details>

<details><summary>Customize</summary>
<p>

To add a custom lr_scheduler create a *my_scheduler.yaml* file in *config/lr_scheduler/*.
A dummy and how this can be used it shown below.
Besides the arguments which are defined in the config, the lr scheduler will be also initialized
with the optimizer, in the following way:
``scheduler=hydra.utils.instantiate(self.config.lr_scheduler.scheduler,
optimizer=self.optimizer,
max_steps=max_steps)``
As you can see also the maximum number of steps is given to the scheduler since this can only be
calculated during runtime.
Even if you do not want to use this information make sure to catch the input argument.

`````yaml
config/lr_scheduler/my_scheduler.yaml
─────────────────────────────
interval: step #or epoch    # when the scheduler should be called, at each step of each epoch
frequency: 1                # how often should it be called, in most cases this should be 1
monitor: metric_to_track    # parameter for pytorch lightning to log the lr
scheduler:                  # defining the actuel scheduler class
  _target_: path.to.my.scheduler.class    # path to your scheduler
  arg1: custom_args        # arguments for the scheduler
  arg2: ...           
`````

````shell
python training.py lr_scheduler=my_scheduler
````

</p>
</details>