# Community Policy

We craft Ludwig with love and care, to the best of our skills and knowledge, and
this project would not be possible without the contribution of an incredible
community.

Members of the Ludwig community provide fixes, new features, functionality, and
documentation, which not only improves Ludwig but also shapes technical
direction.

We are really grateful for that and in exchange we strive to make the
development process as open as possible and communication with the development
team easy and direct.

We strive to create an inclusive community where everyone is respected.
Harassment and any other form of non-inclusive behavior will not be tolerated.

# Issues

If you encounter an issue when using Ludwig, please add it to our
[GitHub Issues tracker](https://github.com/ludwig-ai/ludwig/issues).
Please make sure we are able to replicate the issue by providing the model
definition + command + data or code + data.

If your data cannot be shared, please use the `synthesize_dataset` [command line
utility](../../user_guide/command_line_interface/#synthesize_dataset) to create
a synthetic data with the same feature types.

Example:

```sh
ludwig synthesize_dataset --features="[ \
  {name: text, type: text}, \
  {name: category, type: category}, \
  {name: number, type: number}, \
  {name: binary, type: binary}, \
  {name: set, type: set}, \
  {name: bag, type: bag}, \
  {name: sequence, type: sequence}, \
  {name: timeseries, type: timeseries}, \
  {name: date, type: date}, \
  {name: h3, type: h3}, \
  {name: vector, type: vector}, \
  {name: image, type: image} \
]" --dataset_size=10 --output_path=synthetic_dataset.csv
```

# Forum

We use [GitHub Discussions](https://github.com/ludwig-ai/ludwig/discussions) to
provide a forum for the community to discuss.
Everything that is not an issue and relates Ludwig can be discussed here:
use-cases, requests for help and suggestions, discussions on the future of the
project, and other similar topics. The forum is ideal for asynchronous
communication.

# Chat

We use Slack as a chat solution for allowing both Ludwig users and developers to
interact in a timely, more synchronous way.

[Click here](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)
to receive an invitation.
