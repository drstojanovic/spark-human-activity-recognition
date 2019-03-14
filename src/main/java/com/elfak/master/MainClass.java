/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.elfak.master;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.udf;
import static org.apache.spark.sql.functions.when;
import org.apache.spark.sql.types.DataTypes;

/**
 *
 * @author igord
 */
public class MainClass {

    private static String master = "local[*]";
    public static String ROOT_FOLDER = "D:\\fakultet\\Master rad\\Radovi\\extrasensory_ar_analysis\\data";
    private static String command = "mc";
    private static String label = null;

    private static void processParamteres(String[] args) {
        if (args.length >= 1) {
            master = args[0];
        }
        if (args.length >= 2) {
            ROOT_FOLDER = args[1];
        }
        if (args.length >= 3) {
            command = args[2];
        }
        if (args.length >= 4) {
            label = args[3];
        }
    }

    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     * @throws com.github.sh0nk.matplotlib4j.PythonExecutionException
     */
    public static void main(String[] args) throws IOException, PythonExecutionException {
        processParamteres(args);

        SparkSession spark = SparkSession.builder()
                .appName("ExtraSensoryDemo")
                .master(master)
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        // for spliting all data
        //Dataset<Row> userData = DataRead.readUserData(spark, "1155FF54-63D3-4AB2-9863-8385D0BD0A13");
//        Dataset<Row> userData = DataRead.readAllData(spark);
//        double[] splitWeights = new double[]{0.8, 0.2};
//        Dataset<Row>[] splits = userData.randomSplit(splitWeights);
//        Dataset<Row> train = splits[0].cache();
//        Dataset<Row> test = splits[1].cache();
        Dataset<Row> train = DataRead.readFold(spark, 0, "train").cache();
        Dataset<Row> test = DataRead.readFold(spark, 0, "test").cache();
        // show statistics
        //showUserStatistics(userData);
        //plotUserFeatures(userData);
        // choose sensors and labels
        //String chosenLabel = "LAB_WORK";
//        String[] chosenSensors = new String[]{
//            "raw_acc", "watch_acceleration"
//        };
    
        System.out.println("Spark: " + master);
        System.out.println("FilePath: " + ROOT_FOLDER);
        System.out.println("Command: " + command);
        System.out.println("Label: " + label);

        if ("mc".equals(command)) {
            doMulticlassClassification(train, test);
        } else if (label == null) {
            for (String chosenLabel : DataUtils.allLabels) {
                System.out.println("Using label: " + chosenLabel);
                if (command != null) switch (command) {
                    case "ss":
                        singleSensorClassifier(train, test, chosenLabel);
                        break;
                    case "ef":
                        earlyFusionClassifier(train, test, chosenLabel);
                        break;
                    case "lfa":
                        lateFusionAverageClassifier(train, test, chosenLabel);
                        break;
                    case "lfl":
                        lateFusionWeightsClassifier(train, test, chosenLabel);
                        break;
                    default:
                        break;
                }
            }
        } else {
            String chosenLabel = label;
            
            if (command != null) switch (command) {
                case "ss":
                    singleSensorClassifier(train, test, chosenLabel);
                    break;
                case "ef":
                    earlyFusionClassifier(train, test, chosenLabel);
                    break;
                case "lfa":
                    lateFusionAverageClassifier(train, test, chosenLabel);
                    break;
                case "lfl":
                    lateFusionWeightsClassifier(train, test, chosenLabel);
                    break;
                default:
                    break;
            }
        }
    }

    public static VectorAssembler getSensorFeatureVectorAssembler(
            String[] chosenSensors, String outputColumn) {

        // prepare the list of chosen features
        String[] featureColumns = DataUtils.FeaturesToKeep(DataUtils.allFeatures, chosenSensors);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol(outputColumn);

        return assembler;
    }

    public static Dataset<Row> prepareForLabelClassification(Dataset<Row> data,
            String[] chosenSensors, String label, String featureColumn) {
        return prepareForLabelClassification(data,
                chosenSensors, label, featureColumn, true);
    }

    public static Dataset<Row> prepareForLabelClassification(Dataset<Row> data,
            String[] chosenSensors, String label, String featureColumn, boolean renameLabelColumn) {

        // drop data where there is no classification
        // and rename label column
        Dataset<Row> dataLabeled = data
                .na().drop(new String[]{label});
        if (renameLabelColumn) {
            dataLabeled = dataLabeled.withColumnRenamed(label, "label");
        }

        // prepare the list of chosen features
        String[] featureColumns = DataUtils.FeaturesToKeep(DataUtils.allFeatures, chosenSensors);
        // drop NaNs for chosen sensors
        dataLabeled = dataLabeled
                .na().drop(featureColumns);

        if (dataLabeled.count() < 10) {
            return null;
        }

        // prepare Vectore assembler and apply it
        VectorAssembler assembler = getSensorFeatureVectorAssembler(chosenSensors, featureColumn);
        Dataset<Row> dataAssembled = assembler.transform(dataLabeled);

        return dataAssembled;
    }

    public static void singleSensorClassifier(Dataset<Row> train, Dataset<Row> test,
            String chosenLabel) {

        for (String sensor : DataUtils.basicSensors) {
            System.out.println("--- Using sensor: " + sensor);

            String featureCol = sensor + "Feature";

            // extract only needed data
            Dataset<Row> trainPrepared
                    = prepareForLabelClassification(train, new String[]{sensor}, chosenLabel, featureCol);

            if (trainPrepared == null) {
                System.out.println("Not enough data!");
                continue;
            }

            Dataset<Row> testPrepared
                    = prepareForLabelClassification(test, new String[]{sensor}, chosenLabel, featureCol);

            if (testPrepared == null) {
                System.out.println("Not enough data!");
                continue;
            }

            StandardScaler scaler = new StandardScaler()
                    .setWithMean(true)
                    .setWithStd(true)
                    .setInputCol(featureCol)
                    .setOutputCol("featureVector");
            StandardScalerModel scalerModel = scaler.fit(trainPrepared);
            Dataset<Row> trainData = scalerModel.transform(trainPrepared).cache();
            Dataset<Row> validData = scalerModel.transform(testPrepared).cache();

            //trainData.show(100, false);
            LogisticRegressionModel lrModel = doWeightedLogisticRegression(trainData,
                    "featureVector", "prediction", "probability");

            System.out.println("Train data count: " + trainData.count());
            System.out.println("Test data count: " + validData.count());
            if (lrModel != null) {
                //showMetrics(lrModel, trainData, validData);
                Dataset<Row> validationData = lrModel.transform(validData);
                showMetricsBasic(validationData);
            }

            trainData.unpersist();
            validData.unpersist();
        }
    }

    public static void earlyFusionClassifier(Dataset<Row> train, Dataset<Row> test,
            String chosenLabel) {

        System.out.println("--- Using all sensors:");
        String[] allSensors = DataUtils.basicSensors.toArray(new String[0]);

        String featureCol = "feature";

        // extract only needed data
        Dataset<Row> trainPrepared
                = prepareForLabelClassification(train, allSensors, chosenLabel, featureCol);

        if (trainPrepared == null) {
            System.out.println("Not enough data!");
            return;
        }

        Dataset<Row> testPrepared
                = prepareForLabelClassification(test, allSensors, chosenLabel, featureCol);

        if (testPrepared == null) {
            System.out.println("Not enough data!");
            return;
        }

        StandardScaler scaler = new StandardScaler()
                .setWithMean(true)
                .setWithStd(true)
                .setInputCol(featureCol)
                .setOutputCol("featureVector");
        StandardScalerModel scalerModel = scaler.fit(trainPrepared);
        Dataset<Row> trainData = scalerModel.transform(trainPrepared).cache();
        Dataset<Row> validData = scalerModel.transform(testPrepared).cache();

        LogisticRegressionModel lrModel = doWeightedLogisticRegression(trainData,
                "featureVector", "prediction", "probability");

        System.out.println("Train data count: " + trainData.count());
        System.out.println("Test data count: " + validData.count());

        if (lrModel != null) {
            //showMetrics(lrModel, trainData, validData);
            Dataset<Row> validationData = lrModel.transform(validData);
            showMetricsBasic(validationData);
        }

        trainData.unpersist();
        validData.unpersist();
    }

    public static void lateFusionAverageClassifier(Dataset<Row> trainDataset, Dataset<Row> testDataset,
            String chosenLabel) {

        // cleaning datasets
        // drop not used labels
        String[] labelsToDelete = DataUtils.LabelsToDelete(DataUtils.allLabels, chosenLabel);
        Dataset<Row> train = trainDataset.drop(labelsToDelete);
        Dataset<Row> test = testDataset.drop(labelsToDelete);

        // prepare a list of columns to drop
        String[] featureColumnsToDelete
                = DataUtils.FeaturesToDelete(DataUtils.allFeatures, DataUtils.basicSensors.toArray(new String[0]));
        // keep only chosen features
        train = train.drop(featureColumnsToDelete);
        test = test.drop(featureColumnsToDelete);

        HashMap<String, StandardScalerModel> scalers = new HashMap<>();
        HashMap<String, LogisticRegressionModel> lrModels = new HashMap<>();

        // for training
        for (String sensor : DataUtils.basicSensors) {
            System.out.println("--- Using sensor: " + sensor);

            String featureCol = sensor + "Feature";
            String featureVector = sensor + "FeatureVector";
            String predictionColumn = sensor + "Prediction";
            String probabilityColumn = sensor + "Probability";

            // extract only needed data
            Dataset<Row> trainPrepared
                    = prepareForLabelClassification(train, new String[]{sensor}, chosenLabel, featureCol);

            if (trainPrepared == null) {
                System.out.println("Not enough train data!");
                return;
            }

            StandardScaler scaler = new StandardScaler()
                    .setWithMean(true)
                    .setWithStd(true)
                    .setInputCol(featureCol)
                    .setOutputCol(featureVector);
            StandardScalerModel scalerModel = scaler.fit(trainPrepared);
            scalers.put(sensor, scalerModel);

            Dataset<Row> trainData = scalerModel.transform(trainPrepared).cache();

            LogisticRegressionModel lrModel = doWeightedLogisticRegression(trainData,
                    featureVector, predictionColumn, probabilityColumn);
            lrModels.put(sensor, lrModel);

            System.out.println("Train data count: " + trainData.count());

            trainData.unpersist();
        }

        System.out.println("Train data count: " + train.count());

        // for fitting
        Dataset<Row> testPrepared = test;
        for (String sensor : DataUtils.basicSensors) {
            String featureCol = sensor + "Feature";

            // extract only needed data
            testPrepared
                    = prepareForLabelClassification(testPrepared, new String[]{sensor},
                            chosenLabel, featureCol, false);

            if (testPrepared == null) {
                System.out.println("Not enough test data!");
                return;
            }

            // scale test Data
            StandardScalerModel scalerModel = scalers.get(sensor);
            testPrepared = scalerModel.transform(testPrepared);

            // fit data
            LogisticRegressionModel lrModel = lrModels.get(sensor);
            testPrepared = lrModel.transform(testPrepared);
        }

        System.out.println("Test data count: " + testPrepared.count());

        // remove extra columns
        String[] featureColumnsToKeep
                = DataUtils.FeaturesToKeep(DataUtils.allFeatures, DataUtils.basicSensors.toArray(new String[0]));
        testPrepared = testPrepared
                .drop(featureColumnsToKeep)
                .withColumnRenamed(chosenLabel, "label");

        // sum the probabilities
        int numOfColumns = DataUtils.basicSensors.size();

        UserDefinedFunction first = udf((Vector v) -> v.apply(0), DataTypes.DoubleType);

        String firstColumn = DataUtils.basicSensors.get(0) + "Probability";
        Column sum = first.apply(col(firstColumn));
        for (int i = 1; i < numOfColumns; i++) {
            String columnName = DataUtils.basicSensors.get(i) + "Probability";
            sum = sum.plus(first.apply(col(columnName)));
        }
        Dataset<Row> finalPredicted = testPrepared.withColumn("sumProbabilities", sum);

        // find the average probability
        finalPredicted = finalPredicted.withColumn("avgProbability", col("sumProbabilities").divide(numOfColumns));
        // finally, prediction
        finalPredicted = finalPredicted.withColumn("prediction", when(col("avgProbability").geq(0.5), 0.0).otherwise(1.0));

        showMetricsBasic(finalPredicted);
    }

    public static void lateFusionWeightsClassifier(Dataset<Row> trainDataset, Dataset<Row> testDataset,
            String chosenLabel) {

        // cleaning datasets
        // drop not used labels
        String[] labelsToDelete = DataUtils.LabelsToDelete(DataUtils.allLabels, chosenLabel);
        Dataset<Row> train = trainDataset.drop(labelsToDelete);
        Dataset<Row> test = testDataset.drop(labelsToDelete);

        // prepare a list of columns to drop
        String[] featureColumnsToDelete
                = DataUtils.FeaturesToDelete(DataUtils.allFeatures, DataUtils.basicSensors.toArray(new String[0]));
        // keep only chosen features
        train = train.drop(featureColumnsToDelete);
        test = test.drop(featureColumnsToDelete);

        // split 
        Dataset<Row>[] splits = train.randomSplit(new double[]{0.5, 0.5}, 1234L);
        Dataset<Row> firstTrain = splits[0].cache();
        Dataset<Row> secondTrain = splits[1].cache();

        HashMap<String, StandardScalerModel> scalers = new HashMap<>();
        HashMap<String, LogisticRegressionModel> lrModels = new HashMap<>();

        // for first training
        for (String sensor : DataUtils.basicSensors) {
            System.out.println("--- Using sensor: " + sensor);

            String featureCol = sensor + "Feature";
            String featureVector = sensor + "FeatureVector";
            String predictionColumn = sensor + "Prediction";
            String probabilityColumn = sensor + "Probability";

            // extract only needed data
            Dataset<Row> trainPrepared
                    = prepareForLabelClassification(firstTrain, new String[]{sensor}, chosenLabel, featureCol);

            if (trainPrepared == null) {
                System.out.println("Not enough train data!");
                return;
            }

            StandardScaler scaler = new StandardScaler()
                    .setWithMean(true)
                    .setWithStd(true)
                    .setInputCol(featureCol)
                    .setOutputCol(featureVector);
            StandardScalerModel scalerModel = scaler.fit(trainPrepared);
            scalers.put(sensor, scalerModel);

            Dataset<Row> trainData = scalerModel.transform(trainPrepared).cache();

            LogisticRegressionModel lrModel = doWeightedLogisticRegression(trainData,
                    featureVector, predictionColumn, probabilityColumn);
            lrModels.put(sensor, lrModel);

            System.out.println("Train data count: " + trainData.count());

            trainData.unpersist();
        }

        System.out.println("Train data count: " + firstTrain.count());

        // prepare for second training
        Dataset<Row> testPrepared = secondTrain;
        for (String sensor : DataUtils.basicSensors) {
            String featureCol = sensor + "Feature";

            // extract only needed data
            testPrepared
                    = prepareForLabelClassification(testPrepared, new String[]{sensor},
                            chosenLabel, featureCol, false);

            if (testPrepared == null) {
                System.out.println("Not enough test data!");
                return;
            }

            // scale test Data
            StandardScalerModel scalerModel = scalers.get(sensor);
            testPrepared = scalerModel.transform(testPrepared);

            // fit data
            LogisticRegressionModel lrModel = lrModels.get(sensor);
            testPrepared = lrModel.transform(testPrepared);
        }

        // remove extra columns
        String[] featureColumnsToKeep
                = DataUtils.FeaturesToKeep(DataUtils.allFeatures, DataUtils.basicSensors.toArray(new String[0]));

        // change
        testPrepared = testPrepared
                .drop(featureColumnsToKeep)
                .withColumnRenamed(chosenLabel, "label");

        // sum the probabilities
        int numOfColumns = DataUtils.basicSensors.size();

        UserDefinedFunction first = udf((Vector v) -> v.apply(0), DataTypes.DoubleType);

        List<String> sensorColumns = new ArrayList<>();
        for (int i = 1; i < numOfColumns; i++) {
            String probColumnName = DataUtils.basicSensors.get(i) + "Probability";
            sensorColumns.add(probColumnName);
            String singleColumnName = DataUtils.basicSensors.get(i) + "Single";
            testPrepared = testPrepared
                    .withColumn(singleColumnName, first.apply(col(probColumnName)));
        }

        // assemble vector for second training
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(sensorColumns.toArray(new String[0]))
                .setOutputCol("learnedWeights");
        testPrepared = assembler.transform(testPrepared);

        // scale vector for second training
        StandardScaler scaler = new StandardScaler()
                .setWithMean(true)
                .setWithStd(true)
                .setInputCol("learnedWeights")
                .setOutputCol("learnedWeightsScaled");
        StandardScalerModel scalerModelFinal = scaler.fit(testPrepared);
        testPrepared = scalerModelFinal.transform(testPrepared);

        LogisticRegressionModel lrModelFinal = doWeightedLogisticRegression(testPrepared,
                "learnedWeightsScaled", "prediction", "probability");

        // now to handle test Data :)
        //1. fit first classifier
        Dataset<Row> testFinal = test;

        for (String sensor : DataUtils.basicSensors) {
            String featureCol = sensor + "Feature";

            // extract only needed data
            testFinal
                    = prepareForLabelClassification(testFinal, new String[]{sensor},
                            chosenLabel, featureCol, false);

            if (testFinal == null) {
                System.out.println("Not enough test data!");
                return;
            }

            // scale test Data
            StandardScalerModel scalerModel = scalers.get(sensor);
            testFinal = scalerModel.transform(testFinal);

            // fit data
            LogisticRegressionModel lrModel = lrModels.get(sensor);
            testFinal = lrModel.transform(testFinal);
        }

        // change
        testFinal = testFinal
                .drop(featureColumnsToKeep)
                .withColumnRenamed(chosenLabel, "label");

        for (int i = 1; i < numOfColumns; i++) {
            String probColumnName = DataUtils.basicSensors.get(i) + "Probability";
            String singleColumnName = DataUtils.basicSensors.get(i) + "Single";

            testFinal = testFinal
                    .withColumn(singleColumnName, first.apply(col(probColumnName)));
        }
        testFinal = assembler.transform(testFinal);
        testFinal = scalerModelFinal.transform(testFinal);
        Dataset<Row> finalPredicted = lrModelFinal.transform(testFinal);

        showMetricsBasic(finalPredicted);

        firstTrain.unpersist();
        secondTrain.unpersist();
    }

    public static LogisticRegressionModel doWeightedLogisticRegression(Dataset<Row> dataset,
            String featureColumn, String predictionColumn, String probabilityColumn) {

        // Re-balancing (weighting) of records to be used in the logistic loss objective function
        long datasetSize = dataset.count();
        double positives = dataset.agg(sum(col("label"))).first().getDouble(0);
        double balancingRatio = positives / datasetSize;

        if (positives == 1.0 || positives == 0.0) {
            System.out.println("Cannot do the classification, not both classes are present");
            return null;
        }

        Dataset<Row> weightedDataset
                = dataset.withColumn("classWeightCol",
                        when(col("label").equalTo(0.0), balancingRatio)
                                .otherwise(1.0 - balancingRatio));

        LogisticRegression lr = new LogisticRegression()
                .setFitIntercept(true)
                .setFamily("binomial")
                .setFeaturesCol(featureColumn)
                .setWeightCol("classWeightCol")
                .setLabelCol("label")
                .setPredictionCol(predictionColumn)
                .setRawPredictionCol(predictionColumn + "Raw")
                .setProbabilityCol(probabilityColumn);
        //.setMaxIter(10)
        //.setRegParam(0.3)
        //.setTol(0.1)
        //.setElasticNetParam(0.8)
        LogisticRegressionModel lrModel = lr.fit(weightedDataset);
        return lrModel;

//        // kfold cross validation
//        //
//        //---------------------------------------------------------------------------
//        // find the best param for Logistic Regression
//        BinaryClassificationEvaluator bceval = new BinaryClassificationEvaluator()
//                .setLabelCol("label")
//                .setRawPredictionCol("prediction");
//        
//        //double eval = bceval.evaluate(predicted);
//        //System.out.println(bceval.getMetricName() + ": " + eval);
//
//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid(lr.regParam(), new double[]{0.0, 1e-6, 1e-5, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1})
//                .build();
//
//        CrossValidator cv = new CrossValidator()
//                .setEstimator(lr)
//                .setEvaluator(bceval)
//                .setNumFolds(5)
//                .setEstimatorParamMaps(paramGrid);
//
//        CrossValidatorModel cvmodel = cv.fit(weightedDataset);
//
//        LogisticRegressionModel bestModel = (LogisticRegressionModel) cvmodel.bestModel();
//
//        double bestParam = bestModel.getRegParam();
//        System.out.println("Best RegParam: " + bestParam);
//
//        //double evalBest = bceval.evaluate(bestModel.transform(validData));
//        //System.out.println("Best " + bceval.getMetricName() + ": " + evalBest);
    }

    public static Dataset<Row> prepareData(Dataset<Row> userData, String[] chosenSensors,
            String chosenLabel, String outputColumn) {

        // drop data where there is no classification
        String[] labelsToDelete = DataUtils.LabelsToDelete(DataUtils.allLabels, chosenLabel);
        Dataset<Row> dataWithSingleLabel = userData
                .drop(labelsToDelete)
                .na().drop(new String[]{chosenLabel});

//        if (dataWithSingleLabel.count() < 10) {
//            return null;
//        }
//
//        // prepare a list of columns to drop
//        String[] featureColumnsToDelete
//                = DataUtils.FeaturesToDelete(DataUtils.allFeatures, chosenSensors);
//        // keep only chosen features
//        dataWithSingleLabel = dataWithSingleLabel
//                .drop(featureColumnsToDelete);
        // prepare the list of chosen features
        String[] featureColumns = DataUtils.FeaturesToKeep(DataUtils.allFeatures, chosenSensors);

        // drop where there are not all features
        dataWithSingleLabel = dataWithSingleLabel
                .na().drop(featureColumns);

        if (dataWithSingleLabel.count() < 10) {
            return null;
        }

        // assemble the vector
        VectorAssembler assembler = new VectorAssembler()
                //.setHandleInvalid("skip")
                .setInputCols(featureColumns)
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataWithSingleLabel);

        // create final dataset
        Dataset<Row> prepared = assembledData
                .select("timestamp", "features", chosenLabel)
                .withColumnRenamed(chosenLabel, "label");
        return prepared;
    }

    public static void showUserStatistics(Dataset<Row> data) {
        String uuid = "";
        List<String> allFeatureColumns = DataUtils.allFeatures;
        List<String> allLabelColumns = DataUtils.allLabels;

        System.out.println("User " + uuid + " has " + data.count() + " samples.");
        System.out.println("    with " + allFeatureColumns.size() + " features and "
                + allLabelColumns.size() + " labels.");

        data.summary().show();

        // count labels (missing are also 0)
        String[] labelColumns = allLabelColumns.toArray(new String[0]);
        Dataset<Row> labelCounts = data
                .na().fill(0, labelColumns)
                .groupBy().sum(labelColumns);

        labelCounts.show();

//        //prepare data for export
//        StringBuilder sb = new StringBuilder();
//        Row row = labelCounts.first();
//        for (int i = 0; i < row.size(); i++) {
//            Object number = row.get(i);
//            String label = labelColumns[i];
//
//            sb.append(label);
//            sb.append(",");
//            sb.append(number.toString());
//            sb.append("\n");
//        }
//        
//        System.out.println(sb.toString());
    }

    public static void plotUserFeatures(Dataset<Row> data) throws IOException, PythonExecutionException {

        // features to display
        String[] features = {
            "raw_acc:magnitude_stats:mean",
            "watch_acceleration:3d:mean_z",
            //            "watch_heading:mean_sin",
            //            "location:log_diameter",
            //            "audio_naive:mfcc2:mean",
            //            "audio_naive:mfcc3:mean",
            "discrete:wifi_status:is_not_reachable", //"discrete:time_of_day:between18and24"
        };

        int secondsInDay = 24 * 60 * 60;

        List<Integer> timestamps = data
                .select("timestamp")
                .map(row -> row.getInt(0), Encoders.INT())
                .collectAsList();

        int timestamp0 = timestamps.get(0);

        List<Double> days = timestamps.stream()
                .map(stamp -> (stamp - timestamp0) / (double) secondsInDay)
                .collect(Collectors.toList());

        for (String feature : features) {
            List<Double> featureValues = data
                    .select(feature)
                    .na().fill(0.0)
                    .map(row -> row.getDouble(0), Encoders.DOUBLE())
                    .collectAsList();

            Plot plt = Plot.create();
            plt.plot()
                    .add(days, featureValues)
                    .linestyle("-")
                    .linewidth("0.3");
            plt.xlabel("vreme angazovanja u danima");
            plt.ylabel("vrednost atributa");
            plt.title(feature);
            plt.show();

            // plot histogram
            List<Double> featureValuesHist = data
                    .select(feature)
                    .na().drop()
                    .map(row -> row.getDouble(0), Encoders.DOUBLE())
                    .collectAsList();

            Plot hist = Plot.create();
            hist.hist()
                    .add(featureValuesHist)
                    .rwidth(0.9)
                    .bins(30);
            hist.xlabel("vrednost atributa");
            hist.ylabel("broj uzoraka");
            hist.title(feature);
            hist.show();
        }
    }

    public static void showMetrics(LogisticRegressionModel lrModel, Dataset<Row> trainDataset, Dataset<Row> validDataset) {
        // Print the coefficients and intercept for logistic regression
        //System.out.println("Coefficients: " + lrModel.coefficients());
        //System.out.println("Intercept: " + lrModel.intercept());

        Dataset<Row> trained = lrModel.evaluate(trainDataset).predictions();
        trained.stat().crosstab("label", "prediction").show();

        Dataset<Row> predicted = lrModel.evaluate(validDataset).predictions();
        predicted.stat().crosstab("label", "prediction").show();

        // Training Summary
        //BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();
        //System.out.println("Accurracy: " + trainingSummary.accuracy());
        //System.out.println("Sensitivity(TPR): " + Arrays.toString(trainingSummary.truePositiveRateByLabel()));
        //System.out.println("(FPR): " + Arrays.toString(trainingSummary.falsePositiveRateByLabel()));
        //System.out.println("Precision: " + Arrays.toString(trainingSummary.precisionByLabel()));
        //double accuracy = trainingSummary.accuracy();
        double tp = predicted
                .where(col("prediction").equalTo(1.0).and(col("label").equalTo(1.0)))
                .count();
        double tn = predicted
                .where(col("prediction").equalTo(0.0).and(col("label").equalTo(0.0)))
                .count();
        double fp = predicted
                .where(col("prediction").equalTo(1.0).and(col("label").equalTo(0.0)))
                .count();
        double fn = predicted
                .where(col("prediction").equalTo(0.0).and(col("label").equalTo(1.0)))
                .count();

        double accuracy2 = (tp + tn) / predicted.count();

        double sensitivity = tp / (tp + fn);
        double specificity = tn / (tn + fp);

        double balancedAccuracy = (sensitivity + specificity) / 2;
        double precision = tp / (tp + fp);

        double f1score = (2 * sensitivity * precision) / (sensitivity + precision);

        System.out.println("----------");
        //System.out.println("Accuracy*:         " + accuracy);
        System.out.println("Accuracy*:         " + accuracy2);
        System.out.println("Sensitivity (TPR): " + sensitivity);
        System.out.println("Specificity (TNR): " + specificity);
        System.out.println("Balanced accuracy: " + balancedAccuracy);
        System.out.println("Precision**:       " + precision);
        System.out.println("F1 score:          " + f1score);
        System.out.println("----------");
    }

    public static void showMetricsBasic(Dataset<Row> predicted) {

        //predicted.stat().crosstab("label", "prediction").show();
        double tp = predicted
                .where(col("prediction").equalTo(1.0).and(col("label").equalTo(1.0)))
                .count();
        double tn = predicted
                .where(col("prediction").equalTo(0.0).and(col("label").equalTo(0.0)))
                .count();
        double fp = predicted
                .where(col("prediction").equalTo(1.0).and(col("label").equalTo(0.0)))
                .count();
        double fn = predicted
                .where(col("prediction").equalTo(0.0).and(col("label").equalTo(1.0)))
                .count();

        double accuracy = (tp + tn) / predicted.count();

        double sensitivity = tp / (tp + fn);
        double specificity = tn / (tn + fp);

        double balancedAccuracy = (sensitivity + specificity) / 2;
        double precision = tp / (tp + fp);

        double f1score = (2 * sensitivity * precision) / (sensitivity + precision);

        System.out.println("----------");
        System.out.println("TP:                " + tp);
        System.out.println("TN:                " + tn);
        System.out.println("FP:                " + fp);
        System.out.println("FN:                " + fn);
        System.out.println("----------");
        System.out.println("Accuracy*:         " + accuracy);
        System.out.println("Sensitivity (TPR): " + sensitivity);
        System.out.println("Specificity (TNR): " + specificity);
        System.out.println("Balanced accuracy: " + balancedAccuracy);
        System.out.println("Precision**:       " + precision);
        System.out.println("F1 score:          " + f1score);
        System.out.println("----------");
    }

    private static void doMulticlassClassification(Dataset<Row> train, Dataset<Row> test) {

        final String label = "label";

        Dataset<Row> trainFiltered = prepareForMulticlass(train);

        String[] selectedFeatureColumns = DataUtils.FeaturesToKeep(DataUtils.allFeatures,
                DataUtils.selectedSensors.toArray(new String[0]));
        // assemble the vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatureColumns)
                .setOutputCol("features");
        Dataset<Row> trainAssembled = assembler.transform(trainFiltered);

        // select just features
        Dataset<Row> prepared = trainAssembled.select(col("features"), col(label));

        long totalSum = prepared.count();
        Dataset<Row> summedUp = prepared
                .groupBy(col(label)).count();

        double min = summedUp.agg(min(col("count"))).head().getLong(0);
        System.out.println("Min: " + min);

        summedUp = summedUp.withColumnRenamed(label, "label_e")
                //.withColumn("weightCol", lit(min).divide(col("count")));
                .withColumn("weightCol", lit(1.0).minus(col("count").divide(totalSum)));
        summedUp.show();

        Dataset<Row> joined = prepared = prepared
                .join(summedUp, summedUp.col("label_e").equalTo(prepared.col(label)))
                .drop(col("label_e"));
        joined.show();

        StandardScaler scaler = new StandardScaler()
                .setWithMean(true)
                .setWithStd(true)
                .setInputCol("features")
                .setOutputCol("featureVector");
        StandardScalerModel scalerModel = scaler.fit(prepared);
        prepared = scalerModel.transform(prepared).cache();

        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol("featureVector")
                .setLabelCol(label)
                .setWeightCol("weightCol")
                .setFitIntercept(true)
                .setFamily("multinomial");
        LogisticRegressionModel model = lr.fit(prepared);

        // Make predictions.
        Dataset<Row> predictions = model.transform(prepared);

        // Select (prediction, true label) and compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Acc: " + accuracy);

        Dataset<Row> confusion = predictions.stat().crosstab("label", "prediction");
        confusion.show();

        IndexToString stringer1 = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predString")
                .setLabels(DataUtils.primaryActivities.toArray(new String[0]));

        IndexToString stringer2 = new IndexToString()
                .setInputCol(label)
                .setOutputCol("labelString")
                .setLabels(DataUtils.primaryActivities.toArray(new String[0]));

        predictions = stringer1.transform(predictions);
        predictions = stringer2.transform(predictions);

        predictions.show();

        Dataset<Row> confusion2 = predictions.stat().crosstab("labelString", "predString");
        confusion2.show();

        // test data
        Dataset<Row> testFiltered = prepareForMulticlass(test);
        Dataset<Row> testAssembled = assembler.transform(testFiltered);
        testAssembled = scalerModel.transform(testAssembled).cache();
        Dataset<Row> testPredictions = model.transform(testAssembled);

        accuracy = evaluator.evaluate(testPredictions);
        System.out.println("AccTest: " + accuracy);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(testPredictions);
        System.out.println("F1Test: " + f1);

        evaluator.setMetricName("weightedPrecision");
        double wp = evaluator.evaluate(testPredictions);
        System.out.println("WP: " + wp);

        evaluator.setMetricName("weightedRecall");
        double wr = evaluator.evaluate(testPredictions);
        System.out.println("WR: " + wr);

        Dataset<Row> confusionTest = testPredictions.stat().crosstab("label", "prediction");
        confusionTest.show();

        prepared.unpersist();
        testAssembled.unpersist();
    }

    private static Dataset<Row> prepareForMulticlass(Dataset<Row> data) {
        // fill NaN values with zeros
        System.out.println("pre: " + data.count());
        data = data.na()
                .fill(0.0, DataUtils.primaryActivities.toArray(new String[0]));

        // transform from one-hot encoding to label column
        Column newColumn = when(col(DataUtils.primaryActivities.get(0)).equalTo(1.0), 0);
        for (int i = 1; i < DataUtils.primaryActivities.size(); i++) {
            String column = DataUtils.primaryActivities.get(i);

            newColumn = newColumn.when(col(column).equalTo(1.0), i);
        }
        newColumn = newColumn.otherwise(-1.0);
        data = data.withColumn("label", newColumn);

        // count minuses
        final long minusCount = data.where(col("label").equalTo(-1.0)).count();
        System.out.println("minus: " + minusCount);

        // filter out minuses
        Dataset<Row> filtered = data
                .where(col("label").notEqual(-1.0));
        System.out.println("filtrirano: " + filtered.count());

        // filter out features
        String[] allFeatureColumns = DataUtils.FeaturesToKeep(DataUtils.allFeatures,
                DataUtils.basicSensors.toArray(new String[0]));

        filtered = filtered
                .na().drop(allFeatureColumns);
        System.out.println("filtrirano2: " + filtered.count());

        return filtered;
    }
}
