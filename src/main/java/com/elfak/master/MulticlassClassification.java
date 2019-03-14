/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.elfak.master;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.when;

/**
 *
 * @author igord
 */
public class MulticlassClassification {

    static void doMulticlassClassification(Dataset<Row> train, Dataset<Row> test) {
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
        Dataset<Row> filtered = data.where(col("label").notEqual(-1.0));
        System.out.println("filtrirano: " + filtered.count());
        
        // filter out features
        String[] allFeatureColumns = DataUtils.FeaturesToKeep(DataUtils.allFeatures,
                DataUtils.basicSensors.toArray(new String[0]));
        
        filtered = filtered.na()
                .drop(allFeatureColumns);
        System.out.println("filtrirano2: " + filtered.count());
        
        return filtered;
    }
    
}
