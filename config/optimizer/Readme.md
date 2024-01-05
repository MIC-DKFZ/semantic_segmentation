## Optimizer

<details><summary>Configure</summary>
<p>

Currently [Stochastic Gradient Descent (SGD)](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
and [MadGrad](https://github.com/facebookresearch/madgrad) are only supported optimizers.
Since the pytorch implementation of SGD is used also other parameters of the SGB class, like
nesterov, can be passed (similar for Madgrad):

- **weight_decay:** default = 0.0005
- **momentum:** default = 0.9

````shell
python training.py optimizer=SGD weight_decay=0.0001 momentum=0.8 +optimizer.nesterov=True
python training.py optimizer=MADGRAD

````

</p>
</details>

<details><summary>Customize</summary>
<p>

To add a custom optimizer create a *my_optimizer.yaml* file in *config/optimizer/*.
A dummy and how this can be used it shown below.
Besides the arguments which are defined in the config the optimizer will be also initialized with
the model parameters in the following way:
``optimizer=hydra.utils.instantiate(self.config.optimizer,self.parameters())``

`````yaml
config/optimizer/my_optimizer.yaml
─────────────────────────────
_target_: path.to.my.optimizer.class      # for example torch.optim.SGD
lr: ${lr}
arg1: custom_args
arg2: ...
`````

````shell
python training.py optimizer=my_optimizer
````

</p>
</details>