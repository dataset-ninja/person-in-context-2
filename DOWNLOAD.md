Dataset **Person in Context** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMzU1Ml9QZXJzb24gaW4gQ29udGV4dC9wZXJzb24taW4tY29udGV4dC1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJOY1o2YUJ1U3hXQzRyc3UrclN3a2U0UktvZjMwaWZzUC8rdm1OUWRUMmhzPSJ9?response-content-disposition=attachment%3B%20filename%3D%22person-in-context-DatasetNinja.tar%22)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Person in Context', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://drive.google.com/open?id=1TATEqQwRrlb8ero1yurL1Nd9dIknQnyX).