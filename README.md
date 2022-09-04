# BAYC-Animated-BoredApes: Speaker-Aware Talking-Head Animation

This is the code repository implementing the paper:

> **MakeItTalk: Speaker-Aware Talking-Head Animation**
>
> [Yang Zhou](https://people.umass.edu/~yangzhou), 
> [Xintong Han](http://users.umiacs.umd.edu/~xintong/), 
> [Eli Shechtman](https://research.adobe.com/person/eli-shechtman), 
> [Jose Echevarria](http://www.jiechevarria.com) , 
> [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/), 
> [Dingzeyu Li](https://dingzeyu.li)
>
> SIGGRAPH Asia 2020
>
> **Abstract** We present a method that generates expressive talking-head videos from a single facial image with audio as the only input. In contrast to previous attempts to learn direct mappings from audio to raw pixels for creating talking faces, our method first disentangles the content and speaker information in the input audio signal. The audio content robustly controls the motion of lips and nearby facial regions, while the speaker information determines the specifics of facial expressions and the rest of the talking-head dynamics. Another key component of our method is the prediction of facial landmarks reflecting the speaker-aware dynamics. Based on this intermediate representation, our method works with many portrait images in a single unified framework, including artistic paintings, sketches, 2D cartoon characters,  Japanese mangas, and stylized caricatures.
In addition, our method generalizes well for faces and characters that were not observed during training. We present extensive quantitative and qualitative evaluation of our method, in addition to user studies, demonstrating generated talking-heads of significantly higher quality compared to prior state-of-the-art methods.
>
> [[Project page]](https://people.umass.edu/~yangzhou/MakeItTalk/) 
> [[Paper]](https://people.umass.edu/~yangzhou/MakeItTalk/MakeItTalk_SIGGRAPH_Asia_Final_round-5.pdf) 
> [[Video]](https://www.youtube.com/watch?v=OU6Ctzhpc6s) 
> [[Arxiv]](https://arxiv.org/abs/2004.12992)
> [[Colab Demo]](quick_demo.ipynb)
> [[Colab Demo TDLR]](quick_demo_tdlr.ipynb)

![image](https://user-images.githubusercontent.com/62747193/161796170-daecb8c3-bfd6-4ef6-b106-c0bdb1c4275e.png)


## **Installation:**
1. Create environment and activate it.
```shell
conda create -n makeittalk_env python=3.6
conda activate makeittalk_env
```

2. Install FFMPEG Tool
```shell
sudo apt-get install ffmpeg
```

3. Install all the relevant packages.
```shell
pip install -r requirements.txt
```

4. You don't need wine for this implementation. It's been removed.

Download the following pre-trained models to ```models/``` folder for testing your own animation.

| Model |  Link to the model | 
| :-------------: | :---------------: |
| Voice Conversion  | [Link](https://drive.google.com/file/d/1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x/view?usp=sharing)  |
| Speech Content Module  | [Link](https://drive.google.com/file/d/1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp/view?usp=sharing)  |
| Speaker-aware Module  | [Link](https://drive.google.com/file/d/1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu/view?usp=sharing)  |
| Image2Image Translation Module  | [Link](https://drive.google.com/file/d/1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a/view?usp=sharing)  |

Download pre-trained embedding [[here]](https://drive.google.com/file/d/18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI/view?usp=sharing) and save to `models/dump` folder.

## Samples

https://user-images.githubusercontent.com/62747193/188305082-3616d5c7-a0f3-4bb1-bd0a-7df5ac4de058.mp4


https://user-images.githubusercontent.com/62747193/188305152-ae47f4ca-4799-4bf4-a6de-8a3909dd92a7.mp4


https://user-images.githubusercontent.com/62747193/188305159-7931fd79-d05e-4932-b610-3b75ee88572d.mp4



