package org.heyjiobum.nn.optimizer;

import java.util.ArrayList;
import java.util.List;

public abstract class Optimizer {
    private double learningRate;

    protected List<double[]> parametersBias;
    protected List<double[]> gradientsBias;

    protected List<double[][]> parametersWeights;
    protected List<double[][]> gradientsWeights;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;

        parametersBias = new ArrayList<>();
        gradientsBias = new ArrayList<>();
        parametersWeights = new ArrayList<>();
        gradientsWeights = new ArrayList<>();
    }

    public void apply(double[] bias, double[] biasGradient) {
        parametersBias.add(bias);
        gradientsBias.add(biasGradient);
    }

    public void apply(double[][] weights, double[][] weightsGradient) {
        parametersWeights.add(weights);
        gradientsWeights.add(weightsGradient);
    }

    public void zeroGrad() {
        parametersBias.clear();
        gradientsBias.clear();

        parametersWeights.clear();
        gradientsWeights.clear();
    }

    public double getLearningRate() {
        return learningRate;
    }

    public abstract void clearState();

    public abstract void nextStep();

}