## 2D GAN

[GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) took the world by storm when they first got
published by [Goodfellow et al.](https://www.iangoodfellow.com/slides/2016-12-9-gans.pdf). From generating high
quality [face images](https://generated.photos/faces)
to [generative art](https://en.wikipedia.org/wiki/Generative_art) & music, the applications have soared. Posed as a
zero-sum game between a generator, generating synthetic samples, and a discriminator, which is learning not to be fooled
by real and fake samples. GANs have superseded all previous achievement of deep learning and spawned a new generation of
machine learning research,
including [adversarial attacks](https://venturebeat.com/2021/05/29/adversarial-attacks-in-machine-learning-what-they-are-and-how-to-stop-them/).
GANs have a central property that is often overlooked. Namely that they don't estimate the exact distribution of the
training samples, but do so implicitly i.e `GAN` == `Implicit Generative Model`.

In this coding exercise, you'd be re-building the core components of GANs, namely the generator and discriminator. You
are provided with a
black-box [mixture of gaussian](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
density function `MoG`. For simplicity `MoG` has only two components whose parameters are provided as input argument to
the class object. Your task is to train the generator which can effectively pass for a functional approximation of `MoG`
in fooling the discriminator.

To keep this exercise contained to a few hours. You are free to use building blocks from PyTorch, f.ex linear layers,
convolution layers, non linearities, loss functions and optimizer etc. Please attach your code solution and a pdf/readme
report for the questions below.

### Exercise

1. Build the data loader, that is populate the `__getitem__` function to return samples generated from `MoG`.
2. Design and build your `Generator` and `Discriminator` model architecture. Can you make some assumptions regarding the
   architecture from the multi-model nature of `MoG`?
3. Write an appropriate loss function for training the GAN following the
   official [min-max recipe](https://www.iangoodfellow.com/slides/2016-12-9-gans.pdf).
4. Train the `Discriminator`
5. Train the `Generator`
6. When the training has converged. Generate samples from it and the `MoG` and plot them with real and synthetic label

### Report questions

1. What heuristics you had to apply for training the GAN ?
2. Was the training stable / unstable ? why
3. Will you implementation scale if `MoG` had more components ? why
4. Do you need to scale / normalize the training set to attain convergence ? why
5. Does adding more layers help ? why
6. Does training for longer epocs/iterations help? why