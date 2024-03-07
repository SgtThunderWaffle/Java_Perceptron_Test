package com.yerboi.simpleperceptron;

import java.io.Serializable;
import java.util.Random;

public class Weights implements Serializable {
    
    private static final long serialVersionUID = 1L;
    //each layer contains weights and neurons in that order
    private double[][] weights;
    private int layer;
    private double[] bias;
    private double[][] gradients;
    private double[] biasGradients;
    private Neuron[] nextNeurons;
    private Neuron[] prevNeurons;
    
    public Weights(int layer, Neuron[] prevNeurons, Neuron[] nextNeurons, int weightLowerBound, int weightUpperBound) {
	this.setLayer(layer);
	this.prevNeurons = prevNeurons;
	this.nextNeurons = nextNeurons;
	this.weights = new double[prevNeurons.length][nextNeurons.length];
	this.setGradients(new double[prevNeurons.length][nextNeurons.length]);
	this.setBias(new double[nextNeurons.length]);
	this.setBiasGradients(new double[nextNeurons.length]);
	Random vals = new Random();
	for (int i = 0; i < prevNeurons.length; i++) {
	    for (int j = 0; j < nextNeurons.length; j++) {
		this.weights[i][j] = vals.nextDouble() * (weightUpperBound - weightLowerBound) + weightLowerBound;
	    }
	}
	for (int i = 0; i < nextNeurons.length; i++) {
	    this.bias[i] = (vals.nextDouble() * (weightUpperBound - weightLowerBound) + weightLowerBound)*2;
	}
    }

    private double[][] computeGradients(double[] expected) {
	for (int i = 0; i < gradients.length; i++) {
	    double prevOutput = prevNeurons[i].getOutput();
	    for (int j = 0; j < gradients[i].length; j++) {
		//double localGradient = prevNeurons[j].getErrorOutputGradient(expected) * prevNeurons[j].getOutputInputGradient();
		gradients[i][j] = nextNeurons[j].getErrorOutputGradient(expected) * nextNeurons[j].getOutputInputGradient() * prevOutput;
	    }
	}
	return gradients;
    }
    
    private double[] computeBiasGradients(double[] expected) {
	for (int i = 0; i < nextNeurons.length; i++) {
	    biasGradients[i] = nextNeurons[i].getErrorOutputGradient(expected) * nextNeurons[i].getOutputInputGradient();
	}
	return biasGradients;
    }
    
    //Getter-Setter Methods
    public int getLayer() {
	return layer;
    }

    public void setLayer(int layer) {
	this.layer = layer;
    }

    public double[] getBias() {
	return bias;
    }

    public void setBias(double[] bias) {
	this.bias = bias;
    }

    public double[][] getWeights() {
	return weights;
    }
    
    public void setWeights(double[][] weights) {
	this.weights = weights;
    }
    
    public double[][] getGradients(double[] expected) {
	return computeGradients(expected);
    }
    
    public double[] getBiasGradients(double[] expected) {
	return computeBiasGradients(expected);
    }

    public void setGradients(double[][] gradients) {
	this.gradients = gradients;
    }
    
    public void setBiasGradients(double[] biasGradients) {
	this.biasGradients = biasGradients;
    }

    public Neuron[] getNextNeurons() {
	return nextNeurons;
    }
    
    public void setNextNeurons(Neuron[] nextNeurons) {
	this.nextNeurons = nextNeurons;
    }
    
    public Neuron[] getPrevNeurons() {
	return prevNeurons;
    }
    
    public void setPrevNeurons(Neuron[] prevNeurons) {
	this.prevNeurons = prevNeurons;
    }
    
}
