/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.elfak.master;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author igord
 */
public class DataRead {

    public static List<String> readUsersInFold(int foldNumber, String type, String phone) throws IOException {
        if (MainClass.ROOT_FOLDER.contains("hdfs")) {
            return DataUtils.getFold0AndroidUserIds(type, phone);
        } else {
            Path path = Paths.get(MainClass.ROOT_FOLDER, "cv_5_folds");
            String folderPath = path.toString();

            // ex. "fold_0_test_android_uuids"
            String fileName = "fold_" + foldNumber + "_" + type + "_" + phone + "_uuids.txt";

            Stream<String> stream = Files.lines(Paths.get(folderPath, fileName));
            List<String> list = stream.collect(Collectors.toList());
            return list;
        }
    }

    public static Dataset<Row> readAllData(SparkSession spark) throws IOException {
        Path path = Paths.get(MainClass.ROOT_FOLDER, "features_labels");
        String folderPath = path.toString();

        String[] paths = Files.walk(Paths.get(folderPath)).filter(Files::isRegularFile).map(Path::toString).filter((file) -> file.endsWith(".csv")).collect(Collectors.toList()).toArray(new String[0]);

        return loadFiles(spark, paths);
    }

    public static Dataset<Row> readUserData(SparkSession spark, String uuid) {
        Path path = Paths.get(MainClass.ROOT_FOLDER, "features_labels", uuid + ".features_labels.csv");
        String filePath = path.toString();

        return loadFiles(spark, new String[]{filePath});
    }

    public static Dataset<Row> readMultipleUserData(SparkSession spark, List<String> uuids) {
        List<String> filePaths = new ArrayList<>();
        uuids.forEach((uuid) -> {
            if (MainClass.ROOT_FOLDER.contains("hdfs")) {
                String path = MainClass.ROOT_FOLDER + "/features_labels/" + uuid + ".features_labels.csv";
                filePaths.add(path);
            } else {
                Path path = Paths.get(MainClass.ROOT_FOLDER, "features_labels", uuid + ".features_labels.csv");
                filePaths.add(path.toString());
            }
        });

        return loadFiles(spark, filePaths.toArray(new String[0]));
    }

    public static Dataset<Row> loadFiles(SparkSession spark, String[] paths) {
        // load data from CSV file
        Dataset<Row> data = spark.read()
                .option("inferSchema", false)
                .option("header", true)
                .option("nanValue", "nan")
                .schema(DataUtils.GetSchema())
                .csv(paths);

        // remove "label:" from column name
        for (String columnName : DataUtils.allLabels) {
            String oldLabelName = "label:" + columnName;
            data = data.withColumnRenamed(oldLabelName, columnName).drop(oldLabelName);
        }
        data = data.drop("label_source");

        return data;
    }

    public static Dataset<Row> readFold(SparkSession spark, int foldNumber, String type) throws IOException {
        List<String> allUuids = readUsersInFold(foldNumber, type, "iphone");
        List<String> androidUuids = readUsersInFold(foldNumber, type, "android");

        // concatanate iphone and android users
        allUuids.addAll(androidUuids);

        return readMultipleUserData(spark, allUuids);
    }

}
