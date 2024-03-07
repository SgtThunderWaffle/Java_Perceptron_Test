package com.yerboi.simpleperceptron;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.Stream;

public class Perceptron implements Serializable {

    private static final long serialVersionUID = 1L;
    private int[] layerNeuronCounts;
    private ArrayList<Neuron[]> neurons;
    private ArrayList<Weights> weights;
    private double learningRate;
    private double momentumRate;
    private ArrayList<double[][]> prevWeightChange;
    private ArrayList<double[]> prevBiasChange;

    public Perceptron(int[] layerNeuronCounts, double learningRate, double momentumRate) {
	this.neurons = new ArrayList<Neuron[]>();
	this.layerNeuronCounts = layerNeuronCounts;
	
	Neuron[] layerNeurons = new Neuron[layerNeuronCounts[0]];
	//First loop, adds input neurons
	for (int j = 0; j < layerNeurons.length; j++) {
	    layerNeurons[j] = new Neuron(0, j, 0, null, null, ActivationFunctions.SIGMOID, 0);
	}
	this.neurons.add(layerNeurons);
	
	//Each loops adds a layer of weights and neurons
	this.weights = new ArrayList<Weights>();
	for (int i = 1; i < layerNeuronCounts.length-1; i++) {
	    /**this.weights.add(new Weights(i, this.neurons.get(i-1), null, -1, 1));
	    
	    for (Neuron n: this.neurons.get(i-1)) {
		n.setNextWeights(this.weights.get(i-1));
	    }**/
	    
	    layerNeurons = new Neuron[layerNeuronCounts[i]];
	    for (int j = 0; j < layerNeurons.length; j++) {
		layerNeurons[j] = new Neuron(i, j, 2, null, null, ActivationFunctions.SIGMOID, 0);
	    }
	    this.neurons.add(layerNeurons);
	    
	    this.weights.add(new Weights(i, this.neurons.get(i-1), layerNeurons, 0, 1));
	    for (Neuron n: this.neurons.get(i)) {
		n.setPrevWeights(this.weights.get(i-1));
	    }
	    //this.neurons.add(layerNeurons);
	    for (Neuron n: this.neurons.get(i-1)) {
		n.setNextWeights(this.weights.get(i-1));
	    }
	}
	
	//Final loop, adds output neurons
	layerNeurons = new Neuron[layerNeuronCounts[layerNeuronCounts.length-1]];
	for (int j = 0; j < layerNeurons.length; j++) {
	    layerNeurons[j] = new Neuron(layerNeuronCounts.length-1, j, 1, null, null, ActivationFunctions.SIGMOID, ErrorFunctions.SQUARE_ERROR, 0);
	}
	this.neurons.add(layerNeurons);
	this.weights.add(new Weights(layerNeuronCounts.length-1, this.neurons.get(this.neurons.size()-2), this.neurons.get(this.neurons.size()-1), 0, 1));
	for (Neuron n: this.neurons.get(this.neurons.size()-1)) {
	    n.setPrevWeights(this.weights.get(layerNeuronCounts.length-2));
	}
	for (Neuron n: this.neurons.get(this.neurons.size()-2)) {
	    n.setNextWeights(this.weights.get(layerNeuronCounts.length-2));
	}
	//this.neurons.add(layerNeurons);
	
	this.learningRate = learningRate;
	this.momentumRate = momentumRate;
	
	this.prevWeightChange = new ArrayList<double[][]>();
	this.prevBiasChange = new ArrayList<double[]>();
	for (int i = 0; i < weights.size(); i++) {
	    this.prevWeightChange.add(new double[this.weights.get(i).getPrevNeurons().length][this.weights.get(i).getNextNeurons().length]);
	    this.prevBiasChange.add(new double[this.weights.get(i).getNextNeurons().length]);
	}
    }
    
    //sets input data to network
    public void setInputData(double[] inputData) throws Exception {
	if (inputData.length == neurons.get(0).length) {
	    for (int i = 0; i < neurons.get(0).length; i++) {
		neurons.get(0)[i].setInput(inputData[i]);
	    }
	} else {
	    throw new Exception();
	}
    }
    
    //gets network output
    public double[] getPerceptronOutput() {
	double[] output = new double[neurons.get(neurons.size()-1).length];
	for (int i = 0; i < output.length; i++) {
	    output[i] = neurons.get(neurons.size()-1)[i].getOutput();
	}
	return output;
    }
    
    //does backpropagation
    public void backpropagateError(double[] correctOutputs) throws Exception {
	if (correctOutputs.length == neurons.get(neurons.size()-1).length) {
	    ArrayList<double[][]> gradients = new ArrayList<double[][]>();
	    ArrayList<double[]> biasGradients = new ArrayList<double[]>();
	    for (Neuron neuron: neurons.get(0)) {
		neuron.getErrorOutputGradient(correctOutputs);
	    }
	    for (int i = 0; i < weights.size(); i++) {
		gradients.add(weights.get(i).getGradients(correctOutputs));
		biasGradients.add(weights.get(i).getBiasGradients(correctOutputs));
	    }
	    for (int i = 0; i < weights.size(); i++) {
		double[][] newWeights = weights.get(i).getWeights();
		double[] newBias = weights.get(i).getBias();
 		double[][] gradientSet = gradients.get(i);
 		double[] biasSet = biasGradients.get(i);
 		double[][] prevWeightAdjust = prevWeightChange.get(i);
 		double[] prevBiasAdjust = prevBiasChange.get(i);
		for (int j = 0; j < gradientSet.length; j++) {
		    boolean biasIsSet = false;
		    for (int k = 0; k < gradientSet[j].length; k++) {
			prevWeightAdjust[j][k] = (gradientSet[j][k] * learningRate) + (prevWeightAdjust[j][k] * momentumRate);
			newWeights[j][k] -= prevWeightAdjust[j][k];
			prevWeightChange.set(i, prevWeightAdjust);
			if (!biasIsSet) {
			    prevBiasAdjust[k] = (biasSet[k] * learningRate) + (prevBiasAdjust[k] * momentumRate);
			    newBias[k] -= prevBiasAdjust[k];
			    prevBiasChange.set(i, prevBiasAdjust);
			    biasIsSet = !biasIsSet;
			}
		    }
		    //newBias[j] -= biasSet[j] * learningRate;
		}
		weights.get(i).setWeights(newWeights);
		weights.get(i).setBias(newBias);
	    }
	} else {
	    throw new Exception();
	}
    }
    
    public static void savePerceptron(String filePath, Perceptron perc) {
	try (FileOutputStream fos = new FileOutputStream(filePath); ObjectOutputStream oos = new ObjectOutputStream(fos);) {
	    oos.writeObject(perc);
	} catch (FileNotFoundException e) {
	    e.printStackTrace();
	} catch (IOException e) {
	    e.printStackTrace();
	}
    }
    
    public static Perceptron loadPerceptron(String filePath) {
	Perceptron perc = null;
	try (FileInputStream fis = new FileInputStream(filePath); ObjectInputStream ois = new ObjectInputStream(fis);) {
	    perc = (Perceptron) ois.readObject();
	} catch (FileNotFoundException e) {
	    e.printStackTrace();
	} catch (IOException e) {
	    e.printStackTrace();
	} catch (ClassNotFoundException e) {
	    e.printStackTrace();
	}
	return perc;
    }
    
    public ArrayList<Weights> getWeights() {
	return weights;
    }
    
}
