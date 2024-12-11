package org.heyjiobum.nn;

import org.heyjiobum.nn.data.ImageData;
import org.heyjiobum.nn.layer.Layer;
import org.heyjiobum.nn.layer.LinearLayer;
import org.heyjiobum.nn.metric.Metric;
import org.heyjiobum.nn.optimizer.Optimizer;

import java.util.List;

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
            trainEpoch(trainData, epoch);
            testEpoch(testData, epoch);
        }
    }

    private void trainEpoch(ImageData[][] trainData, int epoch) {
        this.optimizer.clearState();

        for (Metric metric : metrics)
            metric.clearState();

        for (ImageData[] batchImages : trainData) {
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

            double[][] lossGradientBatch = new double[batchSize][];
            for (int j = 0; j < batchSize; j++) {
                for (int k = 0; k < yPredBatch.length; k++) {
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
        }
    }

    private void testEpoch(ImageData[][] trainData, int epoch) {
        for (Metric metric : metrics) {
            metric.clearState();
        }

        for (ImageData[] batchImages : trainData) {
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
        }
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