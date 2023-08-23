# HuggingFaceM4/idefics-9b Cog model

This is an implementation of the [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@demo prompt="What is in this image?"

## Example:

Question: What is in this image? Answer: A dog named Spike.
