# Contributing

Everyone is welcome to contribute, and we value everybody’s contribution. Code is thus not the only
way to help the community. Answering questions, helping others, reaching out and improving the
documentation are immensely valuable contributions as well.

It also helps us if you spread the word: reference the library from blog posts on the awesome
projects it made possible, shout out on Twitter every time it has helped you, or simply star the
repo to say "thank you".

Check out the official [ludwig docs](https://ludwig-ai.github.io/ludwig-docs/) to get oriented
around the codebase, and join the community!

## Open Issues

Issues are listed at: <https://github.com/ludwig-ai/ludwig/issues>

If you would like to work on any of them, make sure it is not already assigned to someone else.

You can self-assign it by commenting on the Issue page with one of the keywords: `#take` or
`#self-assign`.

Work on your self-assigned issue and eventually create a Pull Request.

## Creating Pull Requests

1. Fork the [repository](https://github.com/ludwig-ai/ludwig) by clicking on the "Fork" button on
   the repository's page. This creates a copy of the code under your GitHub user account.

1. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/ludwig.git
   cd ludwig
   git remote add upstream https://github.com/ludwig-ai/ludwig.git
   ```

1. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   *Do not*\* work on the `master` branch.

1. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e .
   ```

   The above command will install only the packages in "requirements.txt" in the developer mode.  If you would like to
   be able to potentially make changes to the overall Ludwig codebase, then use the following command:

   ```bash
   pip install -e .[full]
   ```

   Please note that in certain Shell environments (e.g., the `Z shell`), the dependencies in brackets have to be quoted:

   ```bash
   pip install -e ."[full]"
   ```

   If you do not need access to the entire Ludwig codebase, but just want to be able to run `pytest` on the essential
   functionality, then you would replace the above command with:

   ```bash
   pip install -e .[test]
   ```

   (Please use `pip install -e ."[test]"` where your Shell environment requires quotes around the square brackets.)

   For the full list of the optional dependencies available in Ludwig, please see
   [Installation Guide](https://ludwig.ai/latest/getting_started/installation/) and "setup.py" in the root of the Ludwig
   repository.

1. On MacOS with Apple Silicon, if this installation approach runs into errors, you may need to install the following
   prerequisites:

   ```bash
   brew install cmake libomp
   ```

   This step requires `homebrew` to be installed on your development machine.

1. Install and run `pre-commit`:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

1. Finally, if during the installation of `horovod`, the exception `ModuleNotFoundError: No module named 'packaging'`
   occurs, you may need to disable `horovod` (that means you will need train with another backend, just not `horovod`).
   To do that, edit the file `requirements_extra.txt` and comment out the line that begins with `horovod`.  After that,
   please execute the long `pip install` command given in the previous step.  With these work-around provisions, your
   installation should run to completion successfully.  If you are still having difficulty, please reach out with the
   specifics of your environment in the Ludwig Community
   [Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ).

1. Develop features on your branch.

1. Format your code by running pre-commits so that your newly added files look nice:

   ```bash
   pre-commit run
   ```

   Pre-commits also run automatically when committing.

1. Once you're happy with your changes, make a commit to record your changes locally:

   ```bash
   git add .
   git commit
   ```

   It is a good idea to sync your copy of the code with the original repository regularly. This
   way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

1. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send
   your contribution to the project maintainers for review.

## Other tips

- Add unit tests for any new code you write.
- Make sure tests pass. See the [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/latest/developer_guide/style_guidelines_and_tests/) for more details.

## Attribution

This contributing guideline is adapted from `huggingface`, available at <https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md>.

## Code of Conduct

Please be mindful of and adhere to the Linux Foundation's
[Code of Conduct](https://lfprojects.org/policies/code-of-conduct) when contributing to Ludwig.
