# Talking Heads

## Description
This project is a PyTorch implementation of [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/pdf/1905.08233.pdf). In this paper, a GAN has been designed and trained to generate realistic talking head models from a only a few head-shots (potentially just one) and the face landmarks to reproduce.

The paper explains the architecture of the model, but a lot of details are missing, and no official implementations or trained models have been released. Therefore, we are trying to recreate the model as best as we can, and to explain the missing details here. __**We need your help in order to finish this, please, contribute!**__

## More implementation details
**Read the original paper before you continue reading**

### Dataset
The Talking Heads model was trained using the [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset. This dataset is publicly available, but in order to download it you will need to fill in a Google Docs requesting access. The training set contains ~140k videos of moving head-shots (not all of them are of different people). 

#### Download
The download consists of 9 files of around 30GB each. Downloading directly from their servers is too slow, so we strongly recommend downloading the version that they have in Google Drive, which will download at around 25MB/s. You can easily download them using the [GDrive CLI client](https://github.com/gdrive-org/gdrive). After download, you'll have to concatenate the files together, and then extract them from the zip. Both steps will also take a considerable amount of time given the great size of the download. Make sure you have enough free space on your disk.

#### Landmarks Generation
As the paper explains, landmarks are extracted from the faces using the [Face Alignment](https://github.com/1adrianb/face-alignment) PyTorch library. This library is easy to install using Pip, and easy to use. However, it should be noted that executing the face extraction model __turns autograd off__, so you'll have to turn it back on if you're planning on training a PyTorch model after using the library on the same run. 
The generated landmarks are returned as a vector of coordinates.

#### Preparation
By reading the paper, it seems like they read the videos on the fly as the extract the batches to feed to the model during training. However, this would be very slow considering that for each batch we would have to:
* Read the video (and most videos are split into several segments).
* Select random frames.
* Generate the landmarks using Face Alignment. 
* Transform the landmarks into RGB images of the same format as the video frames, giving the different features a different color.

So for this reason, we created the `dataset` package, which pre-processes all the videos beforehand, generating a file for each video, containing a list of K+1 random frames, and their corresponding landmarks as a vector of coordinates.
These files can be loaded using the `VoxCelebDataset` implemented in that same package, which will return this list, after having converted the landmarks into images and having performed the specified transformations on all frames and landmarks images.

K is the number of frames used to train the Embedder during the meta-training process. Since we are selecting random frames in this step, we should take some things into consideration:
* K should be defined before pre-processing the dataset, which takes a long time as well. So choose wisely. Once you have pre-processed it's very costly to select a greater value.
* The randomly selected frames will remain the same for each epoch, but we still put a different random frame aside for the Generator every time, and we believe that all frames of a video are relatively similar, so this might not have a big negative impact on the training of the Embedder.


The pre-processed dataset will be much smaller in weight that the raw one.

### Network Architecture

#### General
The architecture of the model is a combination of different architectures from previous works. The general structure is taken from the image transformation network of the following paper: [Perceptual Losses for Real-Time Style Transferand Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf) ([code](https://github.com/dxyang/StyleTransfer/blob/master/network.py)). This model is a CNN with three parts:
* Encoder layers, that takes an image and makes it go through convolutions, reducing their size.
* Residual layers, which don't change the size of the data going through them.
* Decoder layers, which takes the reduced images and upscales them back to an image of the same size as the original ones. No transposed convolutions are used, and instead, regular convolutions are used together with an upscaler in them.

The Generator is supposed to have the same structure, while the Embedder and the Discriminator will only have the encoder layers.

The first change that is done to this architecture, is that the encoder and decoder layers are replaced with a downscaling and upscaling residual layers, similar to Figure 15 in [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/pdf/1809.11096.pdf) ([code](https://github.com/AaronLeong/BigGAN-pytorch/blob/master/model_resnet.py)).

In `network.components`, we implemented these residual modules, based on the implementation of the previously mentioned paper. The main networks, however, are located in `network.network`.

There's no information in the paper about how they initialized the different weights of the networks, so we just decided on normal distributions with std 0.2.

#### Embedder
The Embedder is the simplest network of the three: it is for the greatest part just the encoder component of the previously mentioned StyleTransfer architecture. The changes are few:
* Since the Embedder needs to take two inputs: a head-shot and it's landmark, we had to find a way to combine them into one. In the paper it doesn't say anything besides that they "concatenate" them, so it's not clear which approach they took. We considered merging the two images by overlapping the landmarks over the head-shot and concatenating both images on the channel dimension, and ended up choosing the latter approach. Therefore, the Embedder starts by generating a [6, 224, 224] tensor by merging the two 3 channel images.
* This also means that the first Downsample layer will take 6 layers as the input instead of 3.
* In the paper they say that they "set the minimum number of channels in convolutional layers to 64 and the maximum [...] to 512". So we added another downsample layer so that the first one would output 64 channels and the last one 512, while increasing the number by a factor of 2 after each downsampling.
* In the paper they explain that they inserted a self-attention layer at 32x32 spatial resolution in the downsampling sections. The pictures are of size 224x224, so there's no 32x32 resolution, but before the last downsampling, the pictures will be of size 28x28 which is close to 32x32, so we decided to insert this layer there.
* An adaptive max pool layer followed by a ReLU is added at the end in order to perform the "global max pooling" that they mention in the paper.
* We assumed that the Instance Normalization layers were not necessary in this network, but we might be wrong.

#### Generator
The generator is a more complex network, and there's a lot of details not explained in the paper.

The paper explains that the network uses AdaIN layers, but not exactly where. It does say that the downsampling blocks use regular IN layers, so we kept those there, but were unsure about the placement of the AdaIN ones: should they simply replace the IN layers in the upsampling blocks, should they also be used in the middle residual blocks, or should the only be used in the residual blocks and regular IN layers in the upsampling blocks as well? We finally decided to go for the first option, but without much confidence in this choice, since the input of the AdaIN layers isn't clear either.

These normalization layers are supposed to take values from the projected vector psi as input (aside from the output of the previous layer). This projected vector is created by multiplying the embedded vector provided by the Embedder network by a learnable matrix P. However, it is unclear which shape P or psi must have.

Since we placed an AdaIn layer between each upsampling layer, each one of them requires a different shape of input, ranging from [1, 128, ?, ?] to [1, 3, ?, ?]. The approach that we finally took, was to project a one dimensional vector long enough to contain enough unique values for all components of these layers, and then slicing the corresponding portion of it depending on which AdaIN layer it is, and then fold it into the required shape. We are still not sure this is the intended approach, and it doesn't seem like we're using AdaIN layers correctly.


#### Discriminator
The Discriminator is very similar to the Embedder network, with a few differences:
* Before the global max pooling layer, the paper wants to add "an additional residual block [...] which operates at 4×4 spatial resolution". It is unclear which kind of residual block the mean, and why it operates at 4x4, since at that point we're still at 14x14. It could mean that they placed another downsample block that would go from 14x14 to 7x7, but if we're considering a resolution that is a power of 2, that still would mean that they would be operating at a resolution of 8x8. We could force a downsampling from 14x14 directly to 4x4 by tweaking the kernel and stride of the convolution, but we're not sure that's what they meant. So in this case, we simply added another regular residual layer that doesn't change the size of the channels of the tensor.
* In the end, the reality score is calculated by multiplying the discriminator vector by the corresponding column of the learnable W matrix. This operation still doesn't give great results, and the Discriminator returns only ones.

### Loss functions

Finally, the last piece we need in order to train the network is the Loss functions. 

#### Loss CNT
In the paper they calculate the loss of content by comparing the activations of different layers of two image recognition neural networks (VGG19 and VGGFace) when fed a real image and a generated image. In order to calculate this loss, we need to first obtain the trained models. VGG19 is provided by the torchvision library, so obtaining it is trivial. VGGFace on the other side, isn't. So we had to find the trained parameters ([from here](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html)), and then modify the torchvision function that creates the VGG models to adapt it to these weights. This was done in the `network.vgg` package.

At this point we have access to both networks' structure and weights, but we still need to extract the activations of certain intermediate layers, which we do with the `VGG_Activations` module that we implemented in the same package.

#### Loss ADV
The adversary component of the loss function has two components:
 * The realism score produced by the Discriminator when fed a generated image, which needs to be maximized. 
 * The Feature Matching loss as proposed in [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585v2.pdf).

#### Loss MCH
This matching loss is supposed to encourage the columns of the W matrix in the Discriminator to resemble the encodings of the Embedder network. In their experiments, they found that they could still produce good results (especially for one shot talking head models) when not using this component, although by ignoring it, it's not possible to perform fine-tuning. So for now, we've also ignored it, but it's implemented.

#### Loss D
The loss of the Discriminator simply compares the realism score produced when fed a real image, with the generated image using the same landmarks. Since our Discriminator isn't working properly, this loss doesn't change at all for now.