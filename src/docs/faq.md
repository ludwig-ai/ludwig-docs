## What is the Roadmap for the develpment of Ludwig?

We keep an updatet set of tasks ordered by priority in the [GitHub Projects page](https://github.com/uber/ludwig/projects).


## Do you support \[feature | encoder | decoder\] in Ludwig?

The list of encoders for each feature type is specified in the [User Guide](user_guide.md).
We plan to add additional feature types and additional encoders and decoders for all feature types.
Refer to [this question](#what-additional-features-are-you-working-on) for more details.
If you want to help us implementing your favourite feature or model please take a look at the [Developer Guide](developer_guide.md) to see how to contribute.


## Do all datasets need to be loaded in memory?

At the moment it depends on the type of feature: image features can be dynamically loaded from disk from an opened hdf5 file, while other types of features (that usually take need less memory than image ones) are loaded entirely in memory for speed. We plan to add an option to load also other features from disk in future releases and to also support more input file types and more scalable solutions like [Petastorm](https://github.com/uber/Petastorm).


## My data is on \[ GCS | S3 | Azure \], how can I load it?

Ludwig uses Pandas for loading data at the moment (this may change when we move to Petastorm).
This means that if your service provides a mechanism for loading data with a name handler, you can load it.

These name handlers already work:

- [Google Cloud Storage](https://cloud.google.com/storage/): `gs://`. You just have to install `gcsfs` with `pip install gcsfs>=0.2.1` and you will be able to prive paths to Ludwig with the `gs://` name handler.
- [Amazon S3](https://aws.amazon.com/s3/): `s3://`. You just have to install `boto` with `pip install boto` and you will be able to prive paths to Ludwig with the `s3://` name handler.


## What additional features are you working on?

We will prioritize new features depending on the feedback of the community, but we are already planning to add:

- additional image encoders ([DenseNet](https://arxiv.org/abs/1608.06993) and [FractalNet](https://arxiv.org/abs/1605.07648), ResNext and MobileNet).
- image decoding (both image generation by deconvolution and pixel-wise classification for image segmentation).
- time series decoding.
- additional features types (point clouds, nested lists, multi-sentence documents, graphs, videos).
- additional metrics and losses.
- additional data formatters / tokenizers and dataset-specific preprocessing scripts.

We also want to address some of the current limitations:

- currently the full dataset needs to be loaded in memory in order to train a model. Image features already have a way to dynamically read batches of datapoints from disk, and we want to extend this capability to other datatypes.
- a simple user interface in order to provide a live demo capability.
- document lower level functions.
- optimize the data I/O to TensorFlow.

All these are opportunities to get involved in the community and contribute.
Feel free to reach out to us and ask as there are tasks for all levels of experience.


## Who develops Ludwig?

### Main architect and maintainer

[Piero Molino](http://w4nderlu.st) (Stanford University, previously at Uber AI) is the creator, main architect and maintainer

### Co-maintainers

- Travis Addair (Uber), who helped with updating the Horovod integration
- Jim Thompson (Freddie Mac) who contributed the K-Fold cross validation functionality and greatly helped with the TF2 porting

### Early key contributors

- Yaroslav Dudin (Uber) is a key contributor who helped improving the architecture and developing the image feature (among many other contributions)
- Sai Sumanth Miryala (Facebook AI, previously at Uber AI) contributed all the testing, logging and helped polishing.

### Other main contributors

- Yi Shi (Uber) who implemented the time series encoding
- Ankit Jain (Facebook AI, previously at Uber AI) who implemented the bag feature encoding
- Pranav Subramani (Uber) who contributed documentation
- Alex Sergeev (previously at Uber) and Felipe Petroski Such (Uber) who helped with distributed training
- Doug Blank (Comet ML) who contributed the Comte ML integration
- Patrick Von Platen (Hugging Face) who contributed the audio feature
- John Wahba (Stripe), who contributed the serving functionality
- Ivaylo Stefanov (Strypes), who contributed a substantial improvement to the visualization capabilities
- Carlo Grisetti (DS Group), who contributed improvements on the tracking of metrics during training
- Chris Van Pelt (Weights and Biases) and Boris Dayma (Weights and Biases) who contributed the Weights and Biases integration
- Emidio Torre helped with the initial design of the landing page

## How can I cite Ludwig?

Please use this Bibtex:
```
@misc{Molino2019,
  author = {Piero Molino and Yaroslav Dudin and Sai Sumanth Miryala},
  title = {Ludwig: a type-based declarative deep learning toolbox},
  year = {2019},
  eprint = {arXiv:1909.07930},
}
```
