# Java Spark application for Human Activity Recognition

## Console line arguments:

| Argument | Description | Example |
| -------- | ----------- | ------- |
| sparkMaster     | Path to the Spark master. | `local[*]` or `spark://vm-20172018:7077` |
| dataFolder   | Path to the data used in application. | `C:\\data` or `hdfs://vm-20172018:9000/data` |
| command      | Method to use to analyze data: <br> `mc` - multiclass classification, <br> `ss` - single sensor, <br> `ef` - early fusion, <br> `lfa` - late fusion with averaging, <br> `lfl` - late fusion with learned weights | `mc`, `ss`, `ef`, `lfa` or `lfl` |
| label | Label for wich to apply method. If ommited, all labels will be used | `TALKING` or `BICYCLING` |