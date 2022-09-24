# RainGAN
An adversarial training approach to nowcast precipitaion with neural network models. This work utilizes a UNet shape generator with a combination of two loss, pixel-wise supervised loss and dicriminator loss. Ideally, the pixel-wise loss contributes to the low frequency information such as general shape, location, and rainfall intensity, where discriminator contributes to the higher frequency detail such as the rainfall pattern. 

## Author
ZHANG Ziyue (RIKEN Intern, Center of Computational Sciencce)

## Starting Date
2022 Sep. 1st

## Example case
The left prediction is original UNet (see my other repositry for detail), the right prediction refers to the GAN architecutrue result.
![Screenshot](example.png)

