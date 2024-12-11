package org.heyjiobum.nn;

import org.heyjiobum.nn.data.ImageData;
import org.heyjiobum.nn.layer.Layer;
import org.heyjiobum.nn.metric.Metric;
import org.heyjiobum.nn.optimizer.Optimizer;

public class Model {
    private Layer[] layers;
    private Optimizer optimizer;
    private Metric[] metrics;

    public Model(Layer[] layers, Optimizer optimizer, Metric[] metrics) {
        this.layers = layers;
        this.optimizer = optimizer;
        this.metrics = metrics;

        initLayersParams();
    }

    public void fit(ImageData[][] trainData, ImageData[][] testData, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Epoch " + epoch + "/" + epochs);
            trainEpoch(trainData);
            testEpoch(testData);
        }
    }

    private void trainEpoch(ImageData[][] trainData) {
        this.optimizer.clearState();

        for (Metric metric : metrics)
            metric.clearState();

        int batchesSize = trainData.length;
        for (int batchesIdx = 0; batchesIdx < batchesSize; batchesIdx++) {
            ImageData[] batchImages = trainData[batchesIdx];
            int batchSize = batchImages.length;

            double[][] images = new double[batchImages.length][];
            double[][] batchClasses = new double[batchImages.length][];

            for (int i = 0; i < batchSize; i++) {
                ImageData img = batchImages[i];

                images[i] = img.image;
                batchClasses[i] = img.getOneHotClass();
            }

            this.optimizer.zeroGrad();

            double[][] yPredBatch = forward(images);

            int lastLayerNeurons = yPredBatch[0].length;
            double[][] lossGradientBatch = new double[batchSize][lastLayerNeurons];
            for (int j = 0; j < batchSize; j++) {
                for (int k = 0; k < lastLayerNeurons; k++) {
                    lossGradientBatch[j][k] = yPredBatch[j][k] - batchClasses[j][k];
                }
            }

            backward(lossGradientBatch);
            for (int j = 0; j < batchSize; j++) {
                for (Metric metric : metrics) {
                    metric.updateState(yPredBatch[j], batchClasses[j]);
                }
            }

            this.optimizer.nextStep();

            StringBuilder metricsState = new StringBuilder();
            for (Metric metric : metrics)
                metricsState.append(metric.getMetricString()).append(", ");
            System.out.print("\rtrain: " + batchesIdx + "/" + batchesSize + "; " + metricsState);
        }
        System.out.println();
    }

    private void testEpoch(ImageData[][] trainData) {
        for (Metric metric : metrics) {
            metric.clearState();
        }

        int batchesSize = trainData.length;
        for (int batchesIdx = 0; batchesIdx < batchesSize; batchesIdx++) {
            ImageData[] batchImages = trainData[batchesIdx];
            int batchSize = batchImages.length;

            double[][] images = new double[batchImages.length][];
            double[][] batchClasses = new double[batchImages.length][];

            for (int i = 0; i < batchSize; i++) {
                ImageData img = batchImages[i];

                images[i] = img.image;
                batchClasses[i] = img.getOneHotClass();
            }

            double[][] yPredBatch = forward(images);

            for (int j = 0; j < batchSize; j++) {
                for (Metric metric : metrics) {
                    metric.updateState(yPredBatch[j], batchClasses[j]);
                }
            }

            StringBuilder metricsState = new StringBuilder();
            for (Metric metric : metrics)
                metricsState.append(metric.getMetricString()).append(", ");
            System.out.print("\rtest:  " + batchesIdx + "/" + batchesSize + "; " + metricsState);
        }
        System.out.println();
    }

    private double[][] forward(double[][] inBatch) {
        double[][] output = inBatch;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    private void backward(double[][] lossGradientBatch) {
        for (int i = layers.length - 1; i >= 0; i--) {
            Layer layer = layers[i];
            lossGradientBatch = layer.backward(lossGradientBatch, optimizer);
        }
    }

    public void initLayersParams() {
        for (int i = 1; i < layers.length; i++) {
            Layer layer = layers[i];
            Layer prevLayer = layers[i - 1];

            layer.initLayerParams(prevLayer.getNeurons());
        }
    }
}