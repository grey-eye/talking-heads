# Talking Heads

## Description
This project is a PyTorch implementation of [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/pdf/1905.08233.pdf). In this paper, a GAN has been designed and trained to generate realistic talking head models from a only a few head-shots (potentially just one) and the face landmarks to reproduce.

The paper explains the architecture of the model, but a lot of details are missing, and no official implementations or trained models have been released. 

Currently, only the meta training process is implemented and working perfectly. The fine-tuning process will be coming soon!

*Thanks to the community for helping to get this model working!*

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

The pictures form the dataset originally have a resolution of 224x224, but the paper always refer to the resolutions as powers of 2 (4x4, 32x32... etc.), so we decided to scale up the images to 256x256.

#### Embedder
The Embedder is the simplest network of the three: it is for the greatest part just the encoder component of the previously mentioned StyleTransfer architecture. The changes are few:
* Since the Embedder needs to take two inputs: a head-shot and it's landmark, we had to find a way to combine them into one. In the paper it doesn't say anything besides that they "concatenate" them, so it's not clear which approach they took. We considered merging the two images by overlapping the landmarks over the head-shot and concatenating both images on the channel dimension, and ended up choosing the latter approach. Therefore, the Embedder starts by generating a [6, 256, 256] tensor by merging the two 3 channel images.
* This also means that the first Downsample layer will take 6 layers as the input instead of 3.
* In the paper they say that they "set the minimum number of channels in convolutional layers to 64 and the maximum [...] to 512". So we added another downsample layer so that the first one would output 64 channels and the last one 512, while increasing the number by a factor of 2 after each downsampling. They also mention that the smallest resolution they use is 4x4, so we added two more downsample layers which don't add any channels.
* In the paper they explain that they inserted a self-attention layer at 32x32 spatial resolution in the downsampling sections. 
* An adaptive max pool layer followed by a ReLU is added at the end in order to perform the "global max pooling" that they mention in the paper.
* We found out that the Instance Normalization layers are not necessary in this network, although the authors don't mention anything about it.

#### Generator
The generator is a more complex network, and there's a lot of details not explained in the paper.

The paper talks about the use of AdaIN layers, although it's not very clear where they are supposed to be placed. Eventually we understood that they're supposed to replace the BatchNormalization layers *inside the Residual Blocks*, both the regular ones and the upsampling ones. So we made two implementations of these blocks, one using regular instance normalization and one using adaptive normalization.

These normalization layers are supposed to take values from the projected vector psi as input (aside from the output of the previous layer). This projected vector is created by multiplying the embedded vector provided by the Embedder network by a learnable matrix P. However, it is unclear which shape P or psi must have. 
The final shape for psi that we decided on was a vector that would have a length equal to:
* 2 for each "in" channel of each residual layer (one for mean and one for the std of the first AdaIN layer)
* 2 for each "out" channel of each residual layer (one for mean and one for the std of the second AdaIN layer)

As for P, in the paper they name it MLP in a figure, but nowhere else do they mention it being a fully connected layer. In the end, we found out that P works best being a matrix of size [len(psi), 512].

For each Adaptive Residual Layer, we slice the corresponding section of psi to create the parameters for the AdaIN layers inside them.

#### Discriminator
The Discriminator is very similar to the Embedder network, with a few differences:
* An extra regular residual block is added at the end of the decoder,
* In the end, the reality score is calculated by multiplying the discriminator vector by the corresponding column of the learnable W matrix. 
* A sigmoid function is placed at the end to keep the reality score in the range [0, 1]

### Loss functions

Finally, the last piece we need in order to train the network is the Loss functions. 

#### Loss CNT
In the paper they calculate the loss of content by comparing the activations of different layers of two image recognition neural networks (VGG19 and VGGFace) when fed a real image and a generated image. In order to calculate this loss, we need to first obtain the trained models. VGG19 is provided by the torchvision library, so obtaining it is trivial. VGGFace on the other side, isn't. So we had to find the trained parameters ([from here](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html)), and then modify the torchvision function that creates the VGG models to adapt it to these weights. This was done in the `network.vgg` package.

At this point we have access to both networks' structure and weights, but we still need to extract the activations of certain intermediate layers, which we do with the `VGG_Activations` module that we implemented in the same package.

#### Loss ADV
The adversary component of the loss function has two components:
 * The realism score produced by the Discriminator when fed a generated image, which needs to be maximized. 
 * The Feature Matching loss as proposed in [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585v2.pdf). This loss doesn't really make a difference in the end, so we eventually removed it.

#### Loss MCH
This matching loss is supposed to encourage the columns of the W matrix in the Discriminator to resemble the encodings of the Embedder network. In their experiments, they found that they could still produce good results (especially for one shot talking head models) when not using this component, although by ignoring it, it's not possible to perform fine-tuning. 

#### Loss D
The loss of the Discriminator is a hinge loss that compares the realism score produced when fed a real image, with the generated image using the same landmarks. 