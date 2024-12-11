package org.heyjiobum.nn.optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Adam extends Optimizer {
    private double beta1;
    private double beta2;
    private double epsilon;

    private List<double[]> parametersMBias;
    private List<double[]> parametersVBias;

    private List<double[][]> parametersMWeights;
    private List<double[][]> parametersVWeights;

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        this.parametersMBias = new ArrayList<>();
        this.parametersVBias = new ArrayList<>();

        this.parametersMWeights = new ArrayList<>();
        this.parametersVWeights = new ArrayList<>();
    }

    @Override
    public void clearState() {
        parametersMBias.clear();
        parametersVBias.clear();

        parametersMWeights.clear();
        parametersVWeights.clear();
    }

    @Override
    public void nextStep() {
        // Initialize moment vectors for biases and weights if they are empty
        if (parametersMBias.isEmpty() && parametersVBias.isEmpty()) {
            for (double[] param : parametersBias) {
                parametersMBias.add(new double[param.length]);
                parametersVBias.add(new double[param.length]);
            }
        }

        if (parametersMWeights.isEmpty() && parametersVWeights.isEmpty()) {
            for (double[][] weight : parametersWeights) {
                parametersMWeights.add(new double[weight.length][weight[0].length]);
                parametersVWeights.add(new double[weight.length][weight[0].length]);
            }
        }

        // Update bias parameters
        for (int i = 0; i < parametersBias.size(); i++) {
            double[] param = parametersBias.get(i);
            double[] gradient = gradientsBias.get(i);
            double[] m = parametersMBias.get(i);
            double[] v = parametersVBias.get(i);

            // Update biased first moment estimate
            for (int j = 0; j < param.length; j++) {
                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                v[j] = beta2 * v[j] + (1 - beta2) * gradient[j] * gradient[j];
            }

            // Correct bias
            double[] mCorrected = new double[m.length];
            double[] vCorrected = new double[v.length];
            for (int j = 0; j < m.length; j++) {
                mCorrected[j] = m[j] / (1 - beta1);
                vCorrected[j] = v[j] / (1 - beta2);
            }

            // Update parameters
            for (int j = 0; j < param.length; j++) {
                param[j] -= getLearningRate() * mCorrected[j] / (Math.sqrt(vCorrected[j]) + epsilon);
            }
        }

        // Update weight parameters
        for (int i = 0; i < parametersWeights.size(); i++) {
            double[][] weight = parametersWeights.get(i);
            double[][] gradient = gradientsWeights.get(i);
            double[][] m = parametersMWeights.get(i);
            double[][] v = parametersVWeights.get(i);

            // Update biased first moment estimate for weights
            for (int j = 0; j < weight.length; j++) {
                for (int k = 0; k < weight[0].length; k++) {
                    m[j][k] = beta1 * m[j][k] + (1 - beta1) * gradient[j][k];
                    v[j][k] = beta2 * v[j][k] + (1 - beta2) * gradient[j][k] * gradient[j][k];
                }
            }

            // Correct bias for weights
            double[][] mCorrected = new double[m.length][m[0].length];
            double[][] vCorrected = new double[v.length][v[0].length];
            for (int j = 0; j < m.length; j++) {
                for (int k = 0; k < m[0].length; k++) {
                    mCorrected[j][k] = m[j][k] / (1 - beta1);
                    vCorrected[j][k] = v[j][k] / (1 - beta2);
                }
            }

            // Update weights
            for (int j = 0; j < weight.length; j++) {
                for (int k = 0; k < weight[0].length; k++) {
                    weight[j][k] -= getLearningRate() * mCorrected[j][k] / (Math.sqrt(vCorrected[j][k]) + epsilon);
                }
            }
        }
    }
}