## File Formats
Ludwig is able to read UTF-8 encoded data from 14 file formats.
Supported formats are:

- Comma Separated Values (`csv`)
- Excel Workbooks (`excel`)
- Feather (`feather`)
- Fixed Width Format (`fwf`)
- Hierarchical Data Format 5 (`hdf5`)
- Hypertext Markup Language (`html`) Note: limited to single table in the file.
- JavaScript Object Notation (`json` and `jsonl`)
- Parquet (`parquet`)
- Pickled Pandas DataFrame (`pickle`)
- SAS data sets in XPORT or SAS7BDAT format (`sas`)
- SPSS file (`spss`)
- Stata file (`stata`)
- Tab Separated Values (`tsv`)

Ludwig uses Pandas and Dask under the hood to read the UTF-8 encoded dataset files, which allows support for CSV, Excel, Feather, fwf, HDF5, HTML (containing a `<table>`), JSON, JSONL, Parquet, pickle (pickled Pandas DataFrame), SAS, SPSS, Stata and TSV formats.
Ludwig tries to automatically identify the format by the extension.

In case a \*SV file is provided, Ludwig tries to identify the separator (generally `,`) from the data.
The default escape character is `\`.
For example, if `,` is the column separator and one of your data columns has a `,` in it, Pandas would fail to load the data properly.
To handle such cases, we expect the values in the columns to be escaped with backslashes (replace `,` in the data with `\,`).

## Hugging Face Datasets
Ludwig now also supports direct Hugging Face dataset imports with the following syntax (dataset_subset is not always present in Hugging Face datasets, so omit it if necessary).

`"hf://{dataset_name}--{dataset_subset}"`

For example:
`train_stats, _, _ = ludwig_model.train(dataset="hf://mbpp")`
`train_stats, _, _ = ludwig_model.train(dataset="hf://Open-Orca/OpenOrca")`
`train_stats, _, _ = ludwig_model.train(dataset="hf://gsm8k--main")`

Please note that "subset" is not the same as "split". Make sure that you are including the subset name and not the split name when specifying the dataset:

![Alt text](../../images/hf_subset_vs_split.png)