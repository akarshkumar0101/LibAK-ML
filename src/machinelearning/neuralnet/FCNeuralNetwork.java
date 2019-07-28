package machinelearning.neuralnet;

import array.Arrays;
import array.DoubleArrays;
import data.function.DoubleFunction1D;
import math.AKRandom;
import math.matrix.TensorMath;

public class FCNeuralNetwork {

	public final int[] networkDimensions;

	// nodes[layer][nodeid]
	public double[][] nodes;

	// nodes before the sigmoid activation function
	// rawNodes[layer][nodeid]
	public double[][] rawNodes;

	// weights between layer1 and layer2
	// weights[layer1][node2id][node1id]
	public double[][][] weights;

	// biases at layer
	// biases[layer-1][nodeid]
	public double[][] biases;

	private DoubleFunction1D activationFunction = FCNeuralNetwork.sigmoidFunction;
	private DoubleFunction1D activationFunctionDerivative = FCNeuralNetwork.sigmoidFunctionDerivative;

	public FCNeuralNetwork(int... nodesperlayer) {
		this(nodesperlayer, null, null, null);
	}

	public FCNeuralNetwork(double[][][] weights, double[][] biases) {
		this(FCNeuralNetwork.determineNetworkDimensions(weights), null, weights, biases);
	}

	public FCNeuralNetwork(FCNeuralNetwork net) {
		this(net.networkDimensions, net.nodes, net.weights, net.biases);
	}

	public FCNeuralNetwork(Object parameters) {
		this((int[]) ((Object[]) parameters)[0], null, (double[][][]) ((Object[]) parameters)[1],
				(double[][]) ((Object[]) parameters)[2]);
	}

	public FCNeuralNetwork(int[] nodesPerLayer, double[][] nodes, double[][][] weights, double[][] biases) {
		this.networkDimensions = nodesPerLayer.clone();

		this.initNetworkNodes(nodes);
		this.initWeights(weights);
		this.initBiases(biases);
	}

	public DoubleFunction1D getActivationFunction() {
		return this.activationFunction;
	}

	public void setActivationFunction(DoubleFunction1D activationFunction) {
		this.activationFunction = activationFunction;
	}

	public DoubleFunction1D getActivationFunctionDerivative() {
		return this.activationFunctionDerivative;
	}

	public void setActivationFunctionDerivative(DoubleFunction1D activationFunctionDerivative) {
		this.activationFunctionDerivative = activationFunctionDerivative;
	}

	private static int[] determineNetworkDimensions(double[][][] weights) {
		int[] networkDimensions = new int[weights.length + 1];
		for (int i = 0; i < networkDimensions.length - 1; i++) {
			networkDimensions[i] = weights[i][0].length;
		}
		networkDimensions[networkDimensions.length - 1] = Arrays.lastElement(weights).length;
		return networkDimensions;
	}

	public void initNetworkNodes(double[][] nodes) {
		if (nodes != null) {
			this.nodes = DoubleArrays.deepCopy(nodes);
		} else {
			this.nodes = new double[this.networkDimensions.length][];
			for (int l = 0; l < this.networkDimensions.length; l++) {
				this.nodes[l] = new double[this.networkDimensions[l]];
			}
		}
		this.rawNodes = TensorMath.tensorScale(this.nodes, 0);
	}

	public void initWeights(double[][][] weights) {
		if (weights != null) {
			this.weights = DoubleArrays.deepCopy(weights);
		} else {
			this.weights = new double[this.networkDimensions.length - 1][][];
			for (int l = 0; l < this.networkDimensions.length - 1; l++) {
				this.weights[l] = new double[this.networkDimensions[l + 1]][this.networkDimensions[l]];
			}
		}
	}

	public void initBiases(double[][] biases) {
		if (biases != null) {
			this.biases = DoubleArrays.deepCopy(biases);
		} else {
			this.biases = new double[this.networkDimensions.length - 1][];
			for (int l = 1; l < this.networkDimensions.length; l++) {
				this.biases[l - 1] = new double[this.networkDimensions[l]];
			}
		}
	}

	public Object exportParameters() {
		return new Object[] { this.networkDimensions.clone(), DoubleArrays.deepCopy(this.weights),
				DoubleArrays.deepCopy(this.biases) };
	}

	public void feed(double... inputLayerData) {
		if (inputLayerData.length != this.nodes[0].length)
			throw new IllegalArgumentException("Must feed input layer with correct size");
		this.rawNodes[0] = inputLayerData;
		this.nodes[0] = this.rawNodes[0].clone();
	}

	public double[] getOutput() {
		double[] output = Arrays.lastElement(this.nodes);
		return output.clone();
	}

	public static boolean[] activatedNodes(double[] nodes) {
		boolean[] arr = new boolean[nodes.length];
		for (int i = 0; i < nodes.length; i++) {
			arr[i] = Math.floor(nodes[i] + .5) == 1;
		}
		return arr;
	}

	public void randomizeWeightsAndBiases() {
		this.randomizeWeightsAndBiases(-2, 2, -2, 2);
	}

	public void randomizeWeightsAndBiases(double lowlw, double uplw, double lowlb, double uplb) {
		for (double[][] weightmat : this.weights) {
			for (double[] arr : weightmat) {
				for (int i = 0; i < arr.length; i++) {
					double randWeight = AKRandom.randomNumber(lowlw, uplw);
					arr[i] = randWeight;
				}
			}

		}
		for (double[] biasarr : this.biases) {
			for (int i = 0; i < biasarr.length; i++) {
				double randBias = AKRandom.randomNumber(lowlb, uplb);
				biasarr[i] = randBias;
			}
		}
	}

	public void calculateFully() {
		this.calculateLayer(1, true);
	}

	protected void calculateLayer(int layer, boolean recurseFully) {
		if (layer == 0)
			throw new RuntimeException("Cannot calculate input layer");
		if (layer >= this.networkDimensions.length)
			return;

		double[] nodearr = this.nodes[layer - 1];
		double[] biasarr = this.biases[layer - 1];

		double[] result = TensorMath.matrixVectorMult(this.weights[layer - 1], nodearr);
		result = TensorMath.tensorAdd(result, biasarr);
		this.rawNodes[layer] = result;
		this.nodes[layer] = DoubleArrays.performFunction(this.rawNodes[layer], this.activationFunction);

		if (recurseFully) {
			this.calculateLayer(layer + 1, recurseFully);
		}
	}

	@Override
	public String toString() {
		String str = "";
		for (double[] element : this.nodes) {
			for (double element2 : element) {
				str += element2 + " ";
			}
			str += "\n";
		}
		return str;
	}

	public static final DoubleFunction1D sigmoidFunction = inp -> 1 / (1 + Math.exp(-inp));
	public static final DoubleFunction1D sigmoidFunctionDerivative = inp -> FCNeuralNetwork.sigmoidFunction
			.evaluate(inp) * (1 - FCNeuralNetwork.sigmoidFunction.evaluate(inp));
}
