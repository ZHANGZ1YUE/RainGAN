# RainGAN
An adversarial cGAN approach to nowcast precipitaion with neural network models. This work utilizes a UNet shape generator with a Patch GAN discriminator. The objective function is a combination of pixel-wise supervised loss and dicriminator loss. Ideally, the pixel-wise loss contributes to the low frequency information such as general shape, location, and rainfall intensity, where discriminator contributes to the higher frequency detail such as the rainfall pattern. 
(The code is implemented in PyTorch)

## Author
ZHANG Ziyue (UTokyo M2, RIKEN Intern)

## Starting Date
2022 Sep. 1st

## Example case
The left prediction is original UNet, the right prediction refers to the cGAN architecture result.
![Screenshot](example.png)

