Dataset **Person in Context** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzM1NTJfUGVyc29uIGluIENvbnRleHQvcGVyc29uLWluLWNvbnRleHQtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAibGVBbU9EcXBEWTFxd29tbzd6SWhyTDdKSGFjSThGQ0oyV00ySUdNZ3JVRT0ifQ==)

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