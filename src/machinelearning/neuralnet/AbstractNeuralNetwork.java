package machinelearning.neuralnet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import array.Arrays;
import data.tuple.Tuple2D;

public abstract class AbstractNeuralNetwork {

	private final List<Neuron> inputNeurons;
	private final List<Neuron> outputNeurons;

	private final List<Neuron> neurons;

	private final HashMap<Neuron, Tuple2D<HashMap<Neuron, Weight>, Bias>> connections;

	public AbstractNeuralNetwork() {
		this.inputNeurons = new ArrayList<>();
		this.outputNeurons = new ArrayList<>();
		this.neurons = new ArrayList<>();
		this.connections = new HashMap<>();
	}

	public AbstractNeuralNetwork(FCNeuralNetwork fcNetwork) {
		this();
		this.constructFromFCNetwork(fcNetwork);
	}

	private void constructFromFCNetwork(FCNeuralNetwork fcNetwork) {
		double[][] nodes = fcNetwork.nodes;
		Neuron[][] neuronNodes = new Neuron[nodes.length][];
		for (int layer = 0; layer < nodes.length; layer++) {
			neuronNodes[layer] = new Neuron[nodes[layer].length];
			for (int i = 0; i < nodes[layer].length; i++) {
				neuronNodes[layer][i] = new Neuron(nodes[layer][i]);
			}
		}

		// all neurons initialized
		for (Neuron n : neuronNodes[0]) {
			this.inputNeurons.add(n);
		}
		for (Neuron n : Arrays.lastElement(neuronNodes)) {
			this.outputNeurons.add(n);
		}
		for (Neuron[] narr : neuronNodes) {
			for (Neuron n : narr) {
				this.neurons.add(n);
			}
		}

	}

	public HashMap<Neuron, Tuple2D<HashMap<Neuron, Weight>, Bias>> getConnections() {
		return this.connections;
	}

	public abstract boolean isConnected(int neuron1ID, int neuron2ID);

	public abstract double connectionWeight(int neuron1ID, int neuron2ID);
}

class Neuron {

	double value;

	public Neuron(double value) {
		this.value = value;
	}
}

class Weight {
	double value;
}

class Bias {
	double value;
}
