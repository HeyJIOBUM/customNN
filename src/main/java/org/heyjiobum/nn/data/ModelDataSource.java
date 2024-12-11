package org.heyjiobum.nn.data;
import org.heyjiobum.Tuple;

public class ModelDataSource {
    private final Tuple<double[][], int[]> testData;
    private final Tuple<double[][], int[]> trainData;
    private final int batchSize;

    public ModelDataSource(
            Tuple<double[][], int[]> trainData,
            Tuple<double[][], int[]> testData,
            int batchSize) {
        this.trainData = trainData;
        this.testData = testData;
        this.batchSize = batchSize;
    }

/*    public DataBatchWrapper trainDataBatches() {
        return new DataBatchWrapper(trainData, batchSize);
    }*/

/*    public DataBatchWrapper testDataBatches() {
        return new DataBatchWrapper(testData, batchSize);
    }*/
}
