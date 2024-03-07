package com.yerboi.simpleperceptron;

import java.io.Serializable;
import java.util.stream.DoubleStream;

public class Neuron implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private int layer;
    private int id;
    //0 for input, 1 for output, 2 for internal
    private int neuronType;
    private ErrorFunctions errorFunc;
    private Weights prevWeights;
    private Weights nextWeights;
    private ActivationFunctions activFunc;
    private double input;
    private double output;
    
    private double errorOutputGradient;
    private double outputInputGradient;
    
    public Neuron(int layer, int id, int neuronType, Weights prevWeights, Weights nextWeights, ActivationFunctions activFunc, ErrorFunctions errorFunc, double input) {
	this.setLayer(layer);
	this.setId(id);
	this.neuronType = neuronType;
	this.errorFunc = errorFunc;
	this.setPrevWeights(prevWeights);
	this.setNextWeights(nextWeights);
	this.setActivFunc(activFunc);
	this.input = input;
	this.output = -10;
	this.errorOutputGradient = -10;
	this.outputInputGradient = -10;
    }
    
    public Neuron(int layer, int id, int neuronType, Weights prevWeights, Weights nextWeights, ActivationFunctions activFunc, double input) {
	this.setLayer(layer);
	this.setId(id);
	this.neuronType = neuronType;
	this.errorFunc = null;
	this.setPrevWeights(prevWeights);
	this.setNextWeights(nextWeights);
	this.setActivFunc(activFunc);
	this.input = input;
	this.output = -10;
	this.errorOutputGradient = -10;
	this.outputInputGradient = -10;
    }
    
    private double computeOutput() {
	if (neuronType == 0) {
	    setOutput(activFunc.getOutput(input));
	} else {
	    double sum = 0;
	    //double[] prevValues = prevWeights.getWeights()[][id];
	    double[] prevValues = new double[prevWeights.getPrevNeurons().length];
	    for (int i = 0; i < prevValues.length; i++) {
		prevValues[i] = prevWeights.getWeights()[i][id];
	    }
	    for (int i = 0; i < prevWeights.getWeights().length; i++) {
		sum += prevValues[i] * prevWeights.getPrevNeurons()[i].getOutput();
	    }
	    sum += prevWeights.getBias()[id];
	    setOutput(activFunc.getOutput(sum));
	}
	return output;
    }
    
    private double computeErrorOutputGradient(double[] expected) {
	if (neuronType == 1) {
	    errorOutputGradient = errorFunc.getDerivative(getOutput(), expected[id]);
	} else {
	    double sum = 0;
	    for (int i = 0; i < nextWeights.getNextNeurons().length-1; i++) {
		Neuron neuron = nextWeights.getNextNeurons()[i];
		sum += neuron.computeErrorOutputGradient(expected) * neuron.computeOutputInputGradient() * neuron.getPrevWeights().getWeights()[id][i];
	    }
	    /**for (int i = 0; i < nextWeights.getNextNeurons().length; i++) {
		Neuron neuron = nextWeights.getNextNeurons()[i];
		sum += neuron.computeErrorOutputGradient(expected) * neuron.computeOutputInputGradient() * neuron.getPrevWeights().getWeights()[i][id];
	    }**/
	    errorOutputGradient = sum;
	}
	return errorOutputGradient;
    }
    
    private double computeOutputInputGradient() {
	if (neuronType == 0) {
	    outputInputGradient = activFunc.getDerivative(input);
	} else {
	    double[] prevValues = new double[prevWeights.getPrevNeurons().length];
	    for (int i = 0; i < prevValues.length; i++) {
		prevValues[i] = prevWeights.getWeights()[i][id];
	    }
	    outputInputGradient = activFunc.getDerivative(DoubleStream.of(prevValues).sum());
	}
	return outputInputGradient;
    }

    public int getLayer() {
	return layer;
    }

    public void setLayer(int layer) {
	this.layer = layer;
    }
    
    public int getId() {
	return id;
    }
    
    public void setId(int id) {
	this.id = id;
    }
    
    public int getNeuronType() {
	return neuronType;
    }
    
    public void setNeuronType(int neuronType) {
	this.neuronType = neuronType;
    }

    public Weights getPrevWeights() {
	return prevWeights;
    }

    public void setPrevWeights(Weights prevWeights) {
	this.prevWeights = prevWeights;
    }
    
    public Weights getNextWeights() {
	return nextWeights;
    }
    
    public void setNextWeights(Weights nextWeights) {
	this.nextWeights = nextWeights;
    }

    public ActivationFunctions getActivFunc() {
	return activFunc;
    }

    public void setActivFunc(ActivationFunctions activFunc) {
	this.activFunc = activFunc;
    }
    
    public double getInput() {
	return input;
    }
    
    public void setInput(double input) {
	this.input = input;
    }

    public double getOutput() {
	return computeOutput();
    }

    public void setOutput(double output) {
	this.output = output;
    }
    
    public double getErrorOutputGradient(double[] expected) {
	return computeErrorOutputGradient(expected);
    }
    
    public void setErrorOutputGradient(double errorOutputGradient) {
	this.errorOutputGradient = errorOutputGradient;
    }
    
    public double getOutputInputGradient() {
	return computeOutputInputGradient();
    }
    
    public void setOutputInputGradient(double outputInputGradient) {
	this.outputInputGradient = outputInputGradient;
    }
    
}
