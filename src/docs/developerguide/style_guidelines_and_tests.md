Style Guidelines
================

We expect contributions to mimic existing patterns in the codebase and demonstrate good practices: the code should be concise, readable, PEP8-compliant, and conforming to 80 character line length limit.

Tests
=====

We are using ```pytest``` to run tests.
To install all the required dependencies for testing, please do `pip install ludwig[test]`.
Current test coverage is limited to several integration tests which ensure end-to-end functionality but we are planning to expand it.

Checklist
---------

Before running tests, make sure
<br>
1. Your environment is properly setup.<br>
2. You have write access on the machine. Some of the tests require saving data to disk.

Running tests
-------------

To run all tests, just run
```python -m pytest``` from the ludwig root directory.
Note that you don't need to have ludwig module installed and in this case
code change will take effect immediately.

To run a single test, run
``` 
python -m pytest path_to_filename -k "test_method_name"
```

Example
-------

```
python -m pytest tests/integration_tests/test_experiment.py -k "test_visual_question_answering"
```