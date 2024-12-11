package org.heyjiobum.nn.layer;

import org.heyjiobum.nn.activation.*;
import org.heyjiobum.nn.optimizer.*;

public class LinearLayer extends Layer {
    private final Activation activation;

    private double[] bias;
    private double[][] weights;

    private double[][] prevIn;
    private double[][] prevS;
    private double[][] prevOut;

    public LinearLayer(int neurons, Activation activation) {
        super(neurons);
        this.activation = activation;
    }

    @Override
    public double[][] forward(double[][] inBatch) {
        int batchSize = inBatch.length;

        this.prevIn = inBatch;
        this.prevS = new double[batchSize][this.neurons];

        for (int batch = 0; batch < batchSize; batch++) {
            for (int j = 0; j < this.neurons; j++) {
                double[] weightsColumn = getColumn(weights, j);
                prevS[batch][j] = dotProduct(inBatch[batch], weightsColumn) - bias[j];
            }
        }

        this.prevOut = new double[prevS.length][prevS[0].length];
        for (int batch = 0; batch < batchSize; batch++) {
            prevOut[batch] = this.activation.activate(prevS[batch]);
        }

        return prevOut;
    }

    @Override
    public double[][] backward(double[][] deDy, Optimizer optimizer) {
        int batchSize = prevS.length;

        double[][][] dyDs = new double[batchSize][neurons][neurons];
        for (int batch = 0; batch < batchSize; batch++) {
            dyDs[batch] = this.activation.jacobian(prevS[batch]);
        }

        double[][] deDs = new double[batchSize][neurons];
        for (int batch = 0; batch < batchSize; batch++) {
            deDs[batch] = dotProduct(deDy[batch], dyDs[batch]);
        }

        adjustBias(optimizer, deDs);
        adjustPrevWeights(optimizer, deDs);

        double[][] dsDx = transpose(weights);
        double[][] deDx = new double[batchSize][neurons];

        for (int i = 0; i < deDs.length; i++) {
            deDx[i] = dotProduct(deDs[i], dsDx);
        }

        return deDx;
    }

    @Override
    public void initLayerParams(int prevLayerNeurons) {
        this.bias = new double[neurons];
        this.weights = new double[prevLayerNeurons][neurons];

        for (double[] weight : weights) {
            for (int i = 0; i < neurons; i++) {
                weight[i] = Math.random() * Math.sqrt(2.0 / prevLayerNeurons) - Math.sqrt(2.0 / prevLayerNeurons) / 2;
            }
        }
    }

    private void adjustPrevWeights(Optimizer optimizer, double[][] deDs) {
        double[][] weightsGradient = new double[this.weights.length][this.weights[0].length];

        for (int k = 0; k < deDs.length; k++) {
            for (int j = 0; j < prevIn[0].length; j++) {
                for (int l = 0; l < deDs[0].length; l++) {
                    weightsGradient[l][j] += deDs[k][l] * prevIn[k][j];
                }
            }
        }

        optimizer.apply(weights, weightsGradient);
    }

    private void adjustBias(Optimizer optimizer, double[][] deDs) {
        double[] biasGradient = new double[this.neurons];
        for (double[] deDsInBatch : deDs) {
            for (int j = 0; j < neurons; j++) {
                biasGradient[j] -= deDsInBatch[j];
            }
        }

        optimizer.apply(bias, biasGradient);
    }

    private double dotProduct(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private double[] dotProduct(double[] a, double[][] b) {
        double[] result = new double[b[0].length];
        for (int j = 0; j < b[0].length; j++) {
            result[j] = dotProduct(a, getColumn(b, j));
        }
        return result;
    }

    private double[] getColumn(double[][] matrix, int columnIndex) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][columnIndex];
        }
        return column;
    }

    private double[][] transpose(double[][] matrix) {
        double[][] transposed = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }
}