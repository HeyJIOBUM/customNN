package org.heyjiobum.nn.optimizer;

import java.util.ArrayList;
import java.util.List;

public class SGD extends Optimizer {
    public SGD(double learningRate) {
        super(learningRate);
    }

    @Override
    public void clearState() {

    }

    @Override
    public void nextStep() {
        // Update bias parameters
        for (int i = 0; i < parametersBias.size(); i++) {
            double[] param = parametersBias.get(i);
            double[] gradient = gradientsBias.get(i);

            // Update parameters
            for (int j = 0; j < param.length; j++) {
                param[j] -= getLearningRate() * gradient[j];
            }
        }

        // Update weight parameters
        for (int i = 0; i < parametersWeights.size(); i++) {
            double[][] weight = parametersWeights.get(i);
            double[][] gradient = gradientsWeights.get(i);

            // Update weights
            for (int j = 0; j < weight.length; j++) {
                for (int k = 0; k < weight[0].length; k++) {
                    weight[j][k] -= getLearningRate() * gradient[j][k];
                }
            }
        }
    }
}
