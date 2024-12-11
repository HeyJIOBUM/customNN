package org.heyjiobum.nn.layer;

import org.heyjiobum.nn.optimizer.Optimizer;

public abstract class Layer {
    protected int neurons;

    public Layer(int neurons){
        this.neurons = neurons;
    }

    public abstract double[][] forward(double[][] inBatch);

    public abstract double[][] backward(double[][] deDy, Optimizer optimizer);

    public abstract void initLayerParams(int prevLayerNeurons);

    public int getNeurons() {
        return neurons;
    }
}
