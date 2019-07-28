package machinelearning.neuralnet;

import data.tuple.Tuple2D;
import math.AKMath;

//D = Data, A = Answer
public interface OldNeuralNetworkTrainer<D, A> {

	public abstract double[] rawToInputLayer(D d);

	public abstract double[] rawToOutputLayer(A a);

	/**
	 * @return a [? instance of D, ? instance of A];
	 */
	public abstract Tuple2D<D, A> getRandomRawData();

	// return a double such that arr[0] is the input and arr[1] is the
	// expectedOutput
	public default double[][] getRandomTrainingExample() {
		Tuple2D<D, A> da = this.getRandomRawData();
		return new double[][] { this.rawToInputLayer(da.getA()), this.rawToOutputLayer(da.getB()) };
	}

	public default double[][][] getRandomTrainingExamples(int batchSize) {
		double[][][] examples = new double[batchSize][][];
		for (int i = 0; i < examples.length; i++) {
			examples[i] = this.getRandomTrainingExample();
		}
		return examples;
	}

	public static final double STANDARD_ACCEPTABLE_ERROR = .05;

	public default double acceptableOutputError() {
		return OldNeuralNetworkTrainer.STANDARD_ACCEPTABLE_ERROR;
	}

	public default boolean isCorrect(double[] output, double[] expectedOutput) {
		for (int i = 0; i < output.length; i++) {
			if (Math.abs((output[i] - expectedOutput[i]) / expectedOutput[i]) > this.acceptableOutputError())
				return false;
		}
		return true;
	}

	public default double getCost(double[] output, double[] expectedOutput) {
		double cost = 0;
		for (int i = 0; i < output.length; i++) {
			cost += AKMath.sqr(output[i] - expectedOutput[i]);
		}
		return cost;
	}

	public default double getDerivativeOfCost(double output, double expectedOutput) {
		return 2 * (output - expectedOutput);
	}

	public abstract double getWeightLearningRate();

	public default double getBiasLearningRate() {
		return this.getWeightLearningRate();
	}

	public default int getN() {
		return Integer.MAX_VALUE;
	}

	public default void runEveryNIterations(double[] beforePerformance, double[] afterPerformance) {
	}

}
