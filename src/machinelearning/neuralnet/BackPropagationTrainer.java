package machinelearning.neuralnet;

import java.util.List;

import data.tuple.Tuple2D;
import math.matrix.TensorMath;
import program.DetailedRunnable;

public abstract class BackPropagationTrainer<D, A> implements NeuralNetworkTrainer<D, A> {

	private FCNeuralNetwork fcNetwork;

	// these are the partials of the cost function (for given training example) with
	// respect to the following variables

	// nodes[layer][nodeid]
	protected double[][] nodePartials;

	// rawNodePartials is the node values before sigmoid
	protected double[][] rawNodePartials;

	// weights between layer1 and layer2
	// weights[layer1][node2id][node1id]
	protected double[][][] weightPartials;

	// biases at layer
	// biases[layer][nodeid]
	protected double[][] biasPartials;

	public BackPropagationTrainer(FCNeuralNetwork fcNetwork) {
		this.setFcNetwork(fcNetwork);
	}

	public FCNeuralNetwork getFcNetwork() {
		return this.fcNetwork;
	}

	public void setFcNetwork(FCNeuralNetwork fcNetwork) {
		this.fcNetwork = fcNetwork;

		this.nodePartials = TensorMath.tensorScale(fcNetwork.nodes, 0);
		this.rawNodePartials = TensorMath.tensorScale(fcNetwork.rawNodes, 0);
		this.weightPartials = TensorMath.tensorScale(fcNetwork.weights, 0);
		this.biasPartials = TensorMath.tensorScale(fcNetwork.biases, 0);
	}

	public void trainIntensive(int descentIterations, int numExamples) {
		this.trainIntensive(descentIterations, numExamples, null);
	}

	/**
	 * Train Intensive uses gradient descent with the entire training examples
	 * sample size through a given number of iterations, ie proper theoretical
	 * machine learning
	 *
	 * @param descentIterations
	 * @param numExamples
	 * @param toRunEveryDescent runs every descent step in training (optional)
	 */
	public void trainIntensive(int descentIterations, int numExamples, DetailedRunnable toRunEveryDescent) {

		List<Tuple2D<double[], double[]>> examples = this.getRandomTrainingExamples(numExamples);
		for (int i = 0; i < descentIterations; i++) {

			this.gradientDescent(examples);

			if (toRunEveryDescent != null) {
				toRunEveryDescent.run(examples);
			}

		}

	}

	public void trainWithBatches(int descentIterations, int batchSize) {
		this.trainWithBatches(descentIterations, batchSize, null);
	}

	/**
	 * trainWithBatches is a more optimized machine learning technique that uses a
	 * batch of random examples to descend the cost function over and over again,
	 * each iteration is a new batch.
	 *
	 * @param descentIterations
	 * @param batchSize
	 * @param toRunEveryDescent runs every descent step in training (optional)
	 */
	public void trainWithBatches(int descentIterations, int batchSize, DetailedRunnable toRunEveryDescent) {

		for (int i = 0; i < descentIterations; i++) {
			List<Tuple2D<double[], double[]>> examples = this.getRandomTrainingExamples(batchSize);

			this.gradientDescent(examples);

			if (toRunEveryDescent != null) {
				toRunEveryDescent.run(examples);
			}

		}

	}

	protected void gradientDescent(List<Tuple2D<double[], double[]>> examples) {
		double[][][] dW = TensorMath.tensorScale(this.weightPartials, 0);
		double[][] dB = TensorMath.tensorScale(this.biasPartials, 0);
		for (Tuple2D<double[], double[]> example : examples) {
			this.fcNetwork.feed(example.getA());
			this.fcNetwork.calculateFully();
			this.calculatePartials(example.getB(), true);

			dW = TensorMath.tensorAdd(dW, this.weightPartials);
			dB = TensorMath.tensorAdd(dB, this.biasPartials);
		}
		dW = TensorMath.tensorScale(dW, -this.getWeightLearningRate() / 100);
		dB = TensorMath.tensorScale(dB, -this.getBiasLearningRate() / 100);
		this.takeWeightGradientDescentStep(dW);
		this.takeBiasGradientDescentStep(dB);
	}

	protected void calculatePartials(double[] expectedOutput, boolean recurseFully) {
		for (int j = 0; j < this.fcNetwork.networkDimensions[this.fcNetwork.networkDimensions.length - 1]; j++) {
			this.nodePartials[this.fcNetwork.networkDimensions.length - 1][j] = this.getDerivativeOfCost(
					this.fcNetwork.nodes[this.fcNetwork.networkDimensions.length - 1][j], expectedOutput[j]);
		}
		if (recurseFully) {
			this.calculatePartials(this.fcNetwork.networkDimensions.length - 1, recurseFully);
		}
	}

	protected void calculatePartials(int layer, boolean recurseFully) {
		if (layer < 1)
			return;
		for (int i = 0; i < this.fcNetwork.networkDimensions[layer]; i++) {
			this.rawNodePartials[layer][i] = this.nodePartials[layer][i]
					* this.fcNetwork.getActivationFunctionDerivative().evaluate(this.fcNetwork.rawNodes[layer][i]);
			for (int k = 0; k < this.fcNetwork.networkDimensions[layer - 1]; k++) {
				this.weightPartials[layer - 1][i][k] = this.rawNodePartials[layer][i]
						* this.fcNetwork.nodes[layer - 1][k];
			}
			this.biasPartials[layer - 1][i] = this.rawNodePartials[layer][i];
		}
		for (int k = 0; k < this.fcNetwork.networkDimensions[layer - 1]; k++) {
			double sum = 0;
			for (int i = 0; i < this.fcNetwork.networkDimensions[layer]; i++) {
				sum += this.rawNodePartials[layer][i] * this.fcNetwork.weights[layer - 1][i][k];
			}
			this.nodePartials[layer - 1][k] = sum;
		}

		if (recurseFully) {
			this.calculatePartials(layer - 1, recurseFully);
		}
	}

	protected void takeWeightGradientDescentStep(double[][][] dW) {
		this.fcNetwork.weights = TensorMath.tensorAdd(this.fcNetwork.weights, dW);
	}

	protected void takeBiasGradientDescentStep(double[][] dB) {
		this.fcNetwork.biases = TensorMath.tensorAdd(this.fcNetwork.biases, dB);
	}

	/**
	 * trainingExamples[trainingExampleNum][0] is the input layer
	 * trainingExamples[trainingExampleNum][1] is the expected output layer
	 *
	 * @param trainingExamples
	 * @return the average cost of the network and the average accuracy in
	 *         classification
	 */

	public Tuple2D<Double, Double> calculateAveragePerformance(List<Tuple2D<double[], double[]>> trainingExamples) {
		double totalCost = 0, totalAccuracy = 0;
		for (Tuple2D<double[], double[]> example : trainingExamples) {
			Tuple2D<Double, Boolean> performance = this.calculatePerformance(example);
			totalCost += performance.getA();
			totalAccuracy += performance.getB() ? 1 : 0;
		}
		double cost = totalCost / trainingExamples.size();
		double accuracy = totalAccuracy / trainingExamples.size();
		return new Tuple2D<>(cost, accuracy);
	}

	/**
	 * trainingExamples.getA() is the input layer
	 *
	 * trainingExamples.getB() is the expected output layer
	 *
	 * @param trainingExamples
	 * @return the cost of the network ("how bad it is") and boolean if it was
	 *         correct result
	 */
	public Tuple2D<Double, Boolean> calculatePerformance(Tuple2D<double[], double[]> trainingExample) {
		this.fcNetwork.feed(trainingExample.getA());
		this.fcNetwork.calculateFully();
		double[] output = this.fcNetwork.getOutput();
		double[] expectedOutput = trainingExample.getB();

		double cost = this.getCost(output, expectedOutput);
		boolean correct = this.isCorrect(output, expectedOutput);
		return new Tuple2D<>(cost, correct);
	}

}
