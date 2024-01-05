## Environment

<details><summary>Configure</summary>
<p>

If you run code on different devices (e.g. on your local machine and a gpu-cluster) it can make
sense to group all environment specific settings, e.g. paths or hyperparameters like the batch size,
to enable easy switching between them.
Different environments are stored in the *conifg/environment/* folder and can be used in the
following way.
To add you own environment look at the customization chapter. By default ``environment=local``.

````shell
python training.py environment=cluster
python training.py environment=local
````

</p>
 </details>

<details><summary>Customize</summary>
<p>

An environment config contains everything which is specific for the environment like paths or
specific parameters but also to reach environment specific behaviour by for example enable/disable
checkpoint saving or the
progressbar. Since the environment config is merged into the training config at last, you can
override all parameters from there. For adding a new environment config create a *custom_env.yaml*
file in *config/environment/* and adapt the following dummy:

````yaml
config/envrironment/custom_env.yaml
─────────────────────────────
#@package _global_

# Output directory for logs and checkpoints
LOGDIR: logs/
# Paths to datasets
paths:
  cityscapes: /home/.../Datasets/cityscapes
  VOC2010_Context: /home/.../Datasets/VOC2010_Context
  other_datasets: ...
# Whatever you need
CUSTOM_PATH: ...
Some_Parameter: ...
...
````

````shell
python training.py environment=custom_env
````

</p>
</details>