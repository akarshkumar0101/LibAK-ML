package machinelearning.neuralnet;

import array.Arrays;
import array.DoubleArrays;
import data.function.DoubleFunction1D;
import math.matrix.TensorMath;

public class OldNeuralNetwork {

	public final int[] networkDimensions;

	// nodes[layer][nodeid]
	public double[][] nodes;

	// nodes before the sigmoid activation function
	// nodes_Z[layer][nodeid]
	public double[][] nodes_Z;

	// weights between layer1 and layer2
	// weights[layer1][node2id][node1id]
	public double[][][] weights;

	// biases at layer
	// biases[layer-1][nodeid]
	public double[][] biases;

	// these are the partials of the cost function (for given training example) with
	// respect to the following variables

	// nodes[layer][nodeid]
	protected double[][] nodePartials;

	// node_Z_Partials is the node values before sigmoid
	protected double[][] node_Z_Partials;

	// weights between layer1 and layer2
	// weights[layer1][node2id][node1id]
	protected double[][][] weightPartials;

	// biases at layer
	// biases[layer][nodeid]
	protected double[][] biasPartials;

	protected OldNeuralNetworkTrainer<?, ?> trainer;

	public OldNeuralNetwork(int... nodesperlayer) {
		this(nodesperlayer, null, null, null);
	}

	public OldNeuralNetwork(double[][][] weights, double[][] biases) {
		this(OldNeuralNetwork.determineNetworkDimensions(weights), null, weights, biases);
	}

	public OldNeuralNetwork(OldNeuralNetwork net) {
		this(net.networkDimensions, net.nodes, net.weights, net.biases);
	}

	public OldNeuralNetwork(Object parameters) {
		this((int[]) ((Object[]) parameters)[0], null, (double[][][]) ((Object[]) parameters)[1],
				(double[][]) ((Object[]) parameters)[2]);
	}

	public OldNeuralNetwork(int[] nodesPerLayer, double[][] nodes, double[][][] weights, double[][] biases) {
		this.networkDimensions = nodesPerLayer.clone();

		this.initNetworkAndZ(nodes);
		this.initWeights(weights);
		this.initBiases(biases);

		this.initPartials(false);
	}

	private static int[] determineNetworkDimensions(double[][][] weights) {
		int[] networkDimensions = new int[weights.length + 1];
		for (int i = 0; i < networkDimensions.length - 1; i++) {
			networkDimensions[i] = weights[i][0].length;
		}
		networkDimensions[networkDimensions.length - 1] = Arrays.lastElement(weights).length;
		return networkDimensions;
	}

	public void initNetworkAndZ(double[][] nodes) {
		if (nodes != null) {
			this.nodes = DoubleArrays.deepCopy(nodes);
		} else {
			this.nodes = new double[this.networkDimensions.length][];
			for (int l = 0; l < this.networkDimensions.length; l++) {
				this.nodes[l] = new double[this.networkDimensions[l]];
			}
		}
		this.nodes_Z = TensorMath.tensorScale(this.nodes, 0);
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

	protected void initPartials(boolean init) {
		if (init) {
			this.nodePartials = TensorMath.tensorScale(this.nodes, 0);
			this.node_Z_Partials = TensorMath.tensorScale(this.nodes_Z, 0);
			this.weightPartials = TensorMath.tensorScale(this.weights, 0);
			this.biasPartials = TensorMath.tensorScale(this.biases, 0);
		} else {
			this.nodePartials = null;
			this.node_Z_Partials = null;
			this.weightPartials = null;
			this.biasPartials = null;
		}
	}

	public Object exportParameters() {
		return new Object[] { this.networkDimensions.clone(), DoubleArrays.deepCopy(this.weights),
				DoubleArrays.deepCopy(this.biases) };
	}

	public void feed(double... inputLayerData) {
		if (inputLayerData.length != this.nodes[0].length)
			throw new IllegalArgumentException("Must feed input layer with correct size");
		this.nodes[0] = inputLayerData;
		this.nodes_Z[0] = DoubleArrays.deepCopy(inputLayerData);
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
					double randweight = (Math.random() - 0.5) * (uplw - lowlw) + (uplw + lowlw) / 2;
					arr[i] = randweight;
				}
			}

		}
		for (double[] biasarr : this.biases) {
			for (int i = 0; i < biasarr.length; i++) {
				double randbias = (Math.random() - 0.5) * (uplb - lowlb) + (uplb + lowlb) / 2;
				biasarr[i] = randbias;
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

		double[][] result = TensorMath.matrixMult(this.weights[layer - 1], DoubleArrays.toDoubleArray(nodearr));
		result = TensorMath.tensorAdd(result, DoubleArrays.toDoubleArray(biasarr));
		this.nodes_Z[layer] = DoubleArrays.toSingleArray(result);
		this.nodes[layer] = DoubleArrays.performFunction(this.nodes_Z[layer], OldNeuralNetwork.sigmoidFunction);

		if (recurseFully) {
			this.calculateLayer(layer + 1, recurseFully);
		}
	}

	// TRAINING
	public void setTrainer(OldNeuralNetworkTrainer<?, ?> trainer) {
		this.trainer = trainer;
	}

	public OldNeuralNetworkTrainer<?, ?> getTrainer() {
		return this.trainer;
	}

	/**
	 * Train Intensive uses gradient descent with the entire training examples
	 * sample size through a given number of iterations, ie proper theoretical
	 * machine learning
	 *
	 * @param descentIterations
	 * @param numExamples
	 */
	public void trainIntensive(int descentIterations, int numExamples) {
		this.initPartials(true);

		double[][][] examples = this.trainer.getRandomTrainingExamples(numExamples);
		for (int i = 0; i < descentIterations; i++) {

			double[] beforePerformance = null;
			if (i % this.trainer.getN() == 0) {
				beforePerformance = this.calculateAveragePerformance(examples);
			}

			this.gradientDescent(examples);

			if (i % this.trainer.getN() == 0) {
				double[] afterPerformance = this.calculateAveragePerformance(examples);
				this.trainer.runEveryNIterations(beforePerformance, afterPerformance);
			}

		}

		this.initPartials(false);
	}

	/**
	 * trainWithBatches is a more optimized machine learning technique that uses a
	 * batch of random examples to descend the cost function over and over again,
	 * each iteration is a new batch.
	 *
	 * @param descentIterations
	 * @param batchSize
	 */
	public void trainWithBatches(int descentIterations, int batchSize) {
		this.initPartials(true);

		for (int i = 0; i < descentIterations; i++) {
			double[][][] examples = this.trainer.getRandomTrainingExamples(batchSize);

			double[] beforePerformance = null;
			if (i % this.trainer.getN() == 0) {
				beforePerformance = this.calculateAveragePerformance(examples);
			}

			this.gradientDescent(examples);

			if (i % this.trainer.getN() == 0) {
				double[] afterPerformance = this.calculateAveragePerformance(examples);

				this.trainer.runEveryNIterations(beforePerformance, afterPerformance);
			}
		}

		this.initPartials(false);
	}

	protected void gradientDescent(double[][][] examples) {
		double[][][] dW = TensorMath.tensorScale(this.weightPartials, 0);
		double[][] dB = TensorMath.tensorScale(this.biasPartials, 0);
		for (double[][] example : examples) {
			this.feed(example[0]);
			this.calculateFully();
			this.calculatePartials(example[1], true);

			dW = TensorMath.tensorAdd(dW, this.weightPartials);
			dB = TensorMath.tensorAdd(dB, this.biasPartials);
		}
		dW = TensorMath.tensorScale(dW, -this.trainer.getWeightLearningRate() / 100);
		dB = TensorMath.tensorScale(dB, -this.trainer.getBiasLearningRate() / 100);
		this.takeWeightGradientDescentStep(dW);
		this.takeBiasGradientDescentStep(dB);
	}

	protected void calculatePartials(double[] expectedOutput, boolean recurseFully) {
		for (int j = 0; j < this.networkDimensions[this.networkDimensions.length - 1]; j++) {
			this.nodePartials[this.networkDimensions.length - 1][j] = this.trainer
					.getDerivativeOfCost(this.nodes[this.networkDimensions.length - 1][j], expectedOutput[j]);
		}
		if (recurseFully) {
			this.calculatePartials(this.networkDimensions.length - 1, recurseFully);
		}
	}

	protected void calculatePartials(int layer, boolean recurseFully) {
		if (layer < 1)
			return;
		for (int i = 0; i < this.networkDimensions[layer]; i++) {
			this.node_Z_Partials[layer][i] = this.nodePartials[layer][i]
					* OldNeuralNetwork.sigmoidFunctionDerivative.evaluate(this.nodes_Z[layer][i]);
			for (int k = 0; k < this.networkDimensions[layer - 1]; k++) {
				this.weightPartials[layer - 1][i][k] = this.node_Z_Partials[layer][i] * this.nodes[layer - 1][k];
			}
			this.biasPartials[layer - 1][i] = this.node_Z_Partials[layer][i];
		}
		for (int k = 0; k < this.networkDimensions[layer - 1]; k++) {
			double sum = 0;
			for (int i = 0; i < this.networkDimensions[layer]; i++) {
				sum += this.node_Z_Partials[layer][i] * this.weights[layer - 1][i][k];
			}
			this.nodePartials[layer - 1][k] = sum;
		}

		if (recurseFully) {
			this.calculatePartials(layer - 1, recurseFully);
		}
	}

	protected void takeWeightGradientDescentStep(double[][][] dW) {
		this.weights = TensorMath.tensorAdd(this.weights, dW);
	}

	protected void takeBiasGradientDescentStep(double[][] dB) {
		this.biases = TensorMath.tensorAdd(this.biases, dB);
	}

	/**
	 * trainingExamples[trainingExampleNum][0] is the input layer
	 * trainingExamples[trainingExampleNum][1] is the expected output layer
	 *
	 * @param trainingExamples
	 * @return the average cost of the network and the average accuracy in
	 *         classification
	 */

	public double[] calculateAveragePerformance(double[][][] trainingExamples) {
		double totalCost = 0, totalAccuracy = 0;
		for (double[][] example : trainingExamples) {
			Object[] performance = this.calculatePerformance(example);
			totalCost += (double) performance[0];
			totalAccuracy += (boolean) performance[1] ? 1 : 0;
		}
		double cost = totalCost / trainingExamples.length;
		double accuracy = totalAccuracy / trainingExamples.length;
		return new double[] { cost, accuracy };
	}

	/**
	 * trainingExamples[0] is the input layer
	 *
	 * trainingExamples[1] is the expected output layer
	 *
	 * @param trainingExamples
	 * @return the cost of the network ("how bad it is") and boolean if it was
	 *         correct result
	 */
	public Object[] calculatePerformance(double[][] trainingExample) {
		this.feed(trainingExample[0]);
		this.calculateFully();
		double[] output = this.getOutput();
		double[] expectedOutput = trainingExample[1];

		double cost = this.trainer.getCost(output, expectedOutput);
		boolean correct = this.trainer.isCorrect(output, expectedOutput);
		return new Object[] { cost, correct };
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
	public static final DoubleFunction1D sigmoidFunctionDerivative = inp -> OldNeuralNetwork.sigmoidFunction
			.evaluate(inp) * (1 - OldNeuralNetwork.sigmoidFunction.evaluate(inp));
}
